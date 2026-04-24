
const { execSync } = require("child_process");
const path = require("path");
const fs = require("fs");

// 获取 unitorch_microsoft 包的路径
const py = process.env.PYTHON_BIN || "python3";
const pythonPath = execSync(
  `PYTHONWARNINGS=ignore ${py} -c "import os, unitorch_microsoft; print(os.path.dirname(unitorch_microsoft.__file__))"`,
  { stdio: ["pipe", "pipe", "ignore"] }
)
.toString()
.trim()
.split("\n")
.pop();

// 构造 litellm 路径和配置路径
const litellm_config_path = path.join(pythonPath, "configs/litellm/config.yaml");

module.exports = {
  apps: [
    {
      name: "litellm",
      script: "litellm",
      args: `--config ${litellm_config_path}`,
      autorestart: true,
      watch: false,
      interpreter: process.env.PYTHON_BIN
    },
    {
      name: "apps",
      script: "unitorch-fastapi",
      args: "apps/fastapis.ini --device 0",
      autorestart: true,
      watch: false,
      interpreter: process.env.PYTHON_BIN
    }
  ],
};
