# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import re
import ast
import logging
import fire
import pandas as pd


def _read_table(input_file: str, names, escapechar: str) -> pd.DataFrame:
    if isinstance(names, str) and names.strip() == "*":
        names = None
    elif isinstance(names, str):
        names = [n.strip() for n in re.split(r"[,;]", names)]
    return pd.read_csv(
        input_file,
        names=names,
        sep="\t",
        quoting=3,
        header="infer" if names is None else None,
        escapechar=escapechar,
    )


def _last_assignment_name(code: str):
    """Return the variable name of the last assignment statement in code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in reversed(tree.body):
        if isinstance(node, ast.Assign):
            target = node.targets[-1]
            if isinstance(target, ast.Name):
                return target.id
            if isinstance(target, ast.Attribute):
                return target.value.id if isinstance(target.value, ast.Name) else None
    return None


def main(
    output1_file: str = "./output1.txt",
    output2_file: str = "./output2.txt",
    output3_file: str = "./output3.txt",
    input1_file: str = None,
    input2_file: str = None,
    input3_file: str = None,
    input4_file: str = None,
    input5_file: str = None,
    input1_names: str = "*",
    input2_names: str = "*",
    input3_names: str = "*",
    input4_names: str = "*",
    input5_names: str = "*",
    input_escapechar: str = None,
    output_escapechar: str = "\\",
    output_header: bool = False,
    function1: str = "#",
    function2: str = "#",
    function3: str = "#",
    function4: str = "#",
    global_action1: str = "#",
    global_action2: str = "#",
    global_action3: str = "#",
    global_action4: str = "#",
    action1: str = "#",
    action2: str = "#",
    action3: str = "#",
    action4: str = "#",
    action_output1: str = "#",
    action5: str = "#",
    action6: str = "#",
    action_output2: str = "#",
    action_output3: str = "#",
):
    env = {}

    # Load input tables into the execution environment
    for idx, (path, names) in enumerate([
        (input1_file, input1_names),
        (input2_file, input2_names),
        (input3_file, input3_names),
        (input4_file, input4_names),
        (input5_file, input5_names),
    ], start=1):
        if path is not None and os.path.exists(path):
            env[f"Input{idx}"] = _read_table(path, names, input_escapechar)

    # Define helper functions in the environment
    for fn_code in [function1, function2, function3, function4]:
        exec(fn_code.replace("\\n", "\n"), env)
        logging.info(f"Loaded function: `{fn_code}`")

    # Execute global setup actions
    for action in [global_action1, global_action2, global_action3, global_action4]:
        exec(action, env)
        logging.info(f"Global action: `{action}`")

    def _run_actions(actions, output_file):
        output = None
        for action in actions:
            if not action or action.strip() == "#":
                continue
            var_name = _last_assignment_name(action)
            exec(action, env)
            if var_name is not None:
                result = env.get(var_name)
                if result is not None:
                    output = result
                    logging.info(
                        f"Action: `{action}` | Output: {var_name} "
                        f"columns={list(output.columns)} shape={output.shape}"
                    )
        if output is not None:
            output.to_csv(
                output_file,
                sep="\t",
                index=False,
                header=output_header,
                quoting=3,
                escapechar=output_escapechar,
            )
            logging.info(f"Saved output to {output_file}, shape={output.shape}")
        return output

    _run_actions([action1, action2, action3, action4, action_output1], output1_file)
    _run_actions([action5, action6, action_output2], output2_file)
    _run_actions([action_output3], output3_file)


if __name__ == "__main__":
    fire.Fire(main)
