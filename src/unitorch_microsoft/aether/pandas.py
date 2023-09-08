# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import io
import re
import ast
import time
import datetime
import json
import base64
import zipfile
import logging
import hashlib
import numpy as np
import pandas as pd
from unitorch.cli import CoreConfigureParser, GenericScript
from unitorch.cli import register_script
from unitorch_microsoft import cached_path


@register_script("microsoft/script/aether/pandas")
class PandasScript(GenericScript):
    def __init__(self, config: CoreConfigureParser):
        self.config = config

    def launch(self, **kwargs):
        config = self.config
        config.set_default_section("microsoft/script/aether/pandas")
        input1_file = config.getoption("input1_file", None)
        input2_file = config.getoption("input2_file", None)
        input3_file = config.getoption("input3_file", None)
        input4_file = config.getoption("input4_file", None)
        input5_file = config.getoption("input5_file", None)

        input1_names = config.getoption("input1_names", "*")
        input2_names = config.getoption("input2_names", "*")
        input3_names = config.getoption("input3_names", "*")
        input4_names = config.getoption("input4_names", "*")
        input5_names = config.getoption("input5_names", "*")

        input_escapechar = config.getoption("input_escapechar", None)
        output_escapechar = config.getoption("output_escapechar", "\\")

        output_header = config.getoption("output_header", False)

        def get_input_table(input_file, names):
            if isinstance(names, str) and names.strip() == "*":
                names = None
            if isinstance(names, str):
                names = re.split(r"[,;]", names)
                names = [n.strip() for n in names]
            return pd.read_csv(
                input_file,
                names=names,
                sep="\t",
                quoting=3,
                header="infer" if names is None else None,
                escapechar=input_escapechar,
            )

        if input1_file is not None and os.path.exists(input1_file):
            Input1 = get_input_table(input1_file, input1_names)

        if input2_file is not None and os.path.exists(input2_file):
            Input2 = get_input_table(input2_file, input2_names)

        if input3_file is not None and os.path.exists(input3_file):
            Input3 = get_input_table(input3_file, input3_names)

        if input4_file is not None and os.path.exists(input4_file):
            Input4 = get_input_table(input4_file, input4_names)

        if input5_file is not None and os.path.exists(input5_file):
            Input5 = get_input_table(input5_file, input5_names)

        def get_action_output_name(action):
            bodys = ast.parse(action).body
            if len(bodys) == 0:
                return
            for body in reversed(bodys):
                if hasattr(body, "targets"):
                    target = body.targets[-1]
                    if hasattr(target, "id"):
                        return target.id
                    return target.value.id
            return

        output = None

        function1 = config.getoption("function1", "#")
        function2 = config.getoption("function2", "#")
        function3 = config.getoption("function3", "#")
        function4 = config.getoption("function4", "#")

        global_action1 = config.getoption("global_action1", "#")
        global_action2 = config.getoption("global_action2", "#")
        global_action3 = config.getoption("global_action3", "#")
        global_action4 = config.getoption("global_action4", "#")

        for action in [
            function1,
            function2,
            function3,
            function4,
        ]:
            exec(action.replace("\\n", "\n"))
            globals().update(locals())
            logging.info(f"Global Function: `{action}`")

        for action in [
            global_action1,
            global_action2,
            global_action3,
            global_action4,
        ]:
            exec(action)
            globals().update(locals())
            logging.info(f"Global Action: `{action}`")

        action1 = config.getoption("action1", "#")
        action2 = config.getoption("action2", "#")
        action3 = config.getoption("action3", "#")
        action4 = config.getoption("action4", "#")

        action_output1 = config.getoption("action_output1", "#")
        output1_file = config.getoption("output1_file", "./output1.txt")

        for action in [action1, action2, action3, action4, action_output1]:
            if action is not None:
                __output_name__ = get_action_output_name(action)
                exec(action)
                globals().update(locals())
                __output__ = locals().get(__output_name__, None)
                if __output__ is not None:
                    output = __output__
                    logging.info(
                        f"Action: `{action}` | Output Info: Name: {__output_name__} -- {list(output.columns)} -- {output.shape}"
                    )

        if output is not None:
            output.to_csv(
                output1_file,
                sep="\t",
                index=False,
                header=output_header,
                quoting=3,
                escapechar=output_escapechar,
            )
            logging.info(f"Processed Output1 finish. shape is {output.shape}")

        action5 = config.getoption("action5", "#")
        action6 = config.getoption("action6", "#")

        action_output2 = config.getoption("action_output2", "#")
        output2_file = config.getoption("output2_file", "./output2.txt")

        for action in [action5, action6, action_output2]:
            if action is not None:
                __output_name__ = get_action_output_name(action)
                exec(action)
                globals().update(locals())
                __output__ = locals().get(__output_name__, None)
                if __output__ is not None:
                    output = __output__
                    logging.info(
                        f"Action: `{action}` | Output Info: Name: {__output_name__} -- {list(output.columns)} -- {output.shape}"
                    )

        if output is not None:
            output.to_csv(
                output2_file,
                sep="\t",
                index=False,
                header=output_header,
                quoting=3,
                escapechar=output_escapechar,
            )
            logging.info(f"Processed Output2 finish. shape is {output.shape}")

        action_output3 = config.getoption("action_output3", "#")
        output3_file = config.getoption("output3_file", "./output3.txt")

        if action_output3 is not None:
            __output_name__ = get_action_output_name(action_output3)
            exec(action_output3)
            globals().update(locals())
            __output__ = locals().get(__output_name__, None)
            if __output_name__ is not None and __output__ is not None:
                output = __output__
                logging.info(
                    f"Action: `{action}` | Output Info: Name: {__output_name__} -- {list(output.columns)} -- {output.shape}"
                )

        if output is not None:
            output.to_csv(
                output3_file,
                sep="\t",
                index=False,
                header=output_header,
                quoting=3,
                escapechar=output_escapechar,
            )
            logging.info(f"Processed Output3 finish. shape is {output.shape}")
