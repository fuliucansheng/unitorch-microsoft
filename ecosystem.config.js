
const { execSync } = require("child_process");
const path = require("path");
const fs = require("fs");

// 获取 unitorch_microsoft 包的路径
const pythonPath = execSync(
  "python3.11 -c \"import os, unitorch_microsoft; print(os.path.dirname(unitorch_microsoft.__file__))\""
)
.toString()
.trim();

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
    },
    {
      name: "apps",
      script: "unitorch-fastapi",
      args: "apps/fastapis.ini",
      autorestart: true,
      watch: false,
    }
  ],
};
