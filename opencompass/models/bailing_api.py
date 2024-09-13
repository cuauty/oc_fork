import concurrent
import concurrent.futures
import os
import socket
import time
import traceback
from typing import Dict, List, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection

from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]


class HTTPAdapterWithSocketOptions(HTTPAdapter):
    def __init__(self, *args, **kwargs):
        self._socket_options = HTTPConnection.default_socket_options + [
            (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
            (socket.SOL_TCP, socket.TCP_KEEPIDLE, 75),
            (socket.SOL_TCP, socket.TCP_KEEPINTVL, 30),
            (socket.SOL_TCP, socket.TCP_KEEPCNT, 120),
        ]
        super(HTTPAdapterWithSocketOptions, self).__init__(*args, **kwargs)

    def init_poolmanager(self, *args, **kwargs):
        if self._socket_options is not None:
            kwargs["socket_options"] = self._socket_options
        super(HTTPAdapterWithSocketOptions, self).init_poolmanager(*args, **kwargs)


class BaiLingAPI(BaseAPIModel):
    """Model wrapper around BaiLing Service.

    Args:
        ouput_key (str): key for prediction
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        generation_kwargs: other params
        retry (int): Number of retires if the API call fails. Defaults to 2.
    """

    def __init__(
        self,
        token: str,
        path: str,
        query_per_second: int = 1,
        max_seq_len: int = None,
        meta_template: Optional[Dict] = None,
        retry: int = 5,
        generation_kwargs: Dict = {},
    ):
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            query_per_second=query_per_second,
            meta_template=meta_template,
            retry=retry,
            generation_kwargs=generation_kwargs,
        )

        self.logger.info(f"Bailing API Model Init path: {path} ")
        self._sessions = []
        try:
            for i in range(BaiLingAPI.EXECUTOR_NUM):
                adapter = HTTPAdapterWithSocketOptions()
                sess = requests.Session()
                sess.mount("http://", adapter)
                sess.mount("https://", adapter)
                self._sessions.append(sess)
        except:
            self.logger.error("Fail to setup the session. ")
            raise RuntimeError("Fail to setup the session. ")

    EXECUTOR_NUM: int = 3

    def generate(
        self,
        inputs: Union[List[str], PromptList],
        max_out_len: int = 512,
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (Union[List[str], PromptList]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass' API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=BaiLingAPI.EXECUTOR_NUM
        ) as executor:
            future_to_messages = {
                executor.submit(
                    send_request, [data], i, conversation_name, service_url
                ): i
                for i, data in enumerate(total_data_list[:total_test_num])
            }
            for future in concurrent.futures.as_completed(future_to_messages):
                m = future_to_messages[future]
                result = future.result()
                durations.append(result[1])
                finished_test_num += 1
                succ_test_num += 1 if result[2] else 0
                # if finished_test_num % 10 == 0:
                if finished_test_num % 10 == 0:
                    print(f"====={m}: duration={result[1]} result={result[0]}")
        end_point = time.perf_counter()

        with ThreadPoolExecutor(1) as executor:
            results = list(
                executor.map(self._generate, inputs, [max_out_len] * len(inputs))
            )
        self.flush()
        return results

    def bailing_infer(
        self,
        query,
    ):
        try:
            chat_completion = self.client.chat.completions.create(
                model=self.path,
                messages=query,
                temperature=0.4,
                code={"max_output_len": 8192},
                maths={
                    "out_seq_length": 8192,
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "do_sample": False,
                },
            )
            return chat_completion.choices[0].message.content, chat_completion.id

        except Exception as e:
            self.logger.error(
                "infer error, query=%s; model_name=%s;  error=%s",
                query,
                self.path,
                traceback.format_exc(),
            )
            raise e

    def _generate(
        self,
        input: Union[str, PromptList],
        max_out_len: int = 512,
    ) -> str:
        """Generate results given an input.

        Args:
            inputs (str or PromptList): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            str: The generated string.
        """
        assert isinstance(input, (str, PromptList))

        if isinstance(input, str):
            messages = [{"role": "user", "content": input}]
        else:
            messages = []
            for item in input:
                content = item["prompt"]
                if not content:
                    continue
                msg = {"content": content}
                if item["role"] == "HUMAN":
                    msg["role"] = "user"
                elif item["role"] == "BOT":
                    msg["role"] = "assistant"
                elif item["role"] == "SYSTEM":
                    msg["role"] = "system"
                messages.append(msg)

        max_num_retries = 0
        while max_num_retries < self.retry:
            try:

                response, bailing_trace_id = self.bailing_infer(
                    messages,
                )
                self.logger.info(
                    f"input: {messages}, response: {response}, bailing_trace_id: {bailing_trace_id}"
                )
                return response

            except Exception as err:
                print("Request Error:{}".format(err), flush=True)
                time.sleep(1)
                max_num_retries += 1
                continue

        return ""
