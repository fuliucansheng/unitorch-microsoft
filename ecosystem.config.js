
const { execSync } = require("child_process");
const path = require("path");

// 获取 unitorch_microsoft 包的路径（用特殊标记包裹输出，再用正则提取）
const py = process.env.PYTHON_BIN || "python3";
const raw = execSync(
  `PYTHONWARNINGS=ignore ${py} -c "import os, unitorch_microsoft; print('##PATH##' + os.path.dirname(unitorch_microsoft.__file__) + '##PATH##')"`,
  { stdio: ["pipe", "pipe", "ignore"] }
).toString();
const match = raw.match(/##PATH##(.+?)##PATH##/);
if (!match) throw new Error("无法获取 unitorch_microsoft 路径");
const pythonPath = match[1].trim();

// 构造 litellm 路径和配置路径
const litellm_config_path = path.join(pythonPath, "configs/litellm/config.yaml");

const appsDir = path.join(__dirname, "apps");

const webApps = ["spaces", "studios"].map((name) => ({
  name: `web-${name}`,
  script: "pnpm",
  args: "run dev",
  cwd: path.join(appsDir, name),
  autorestart: true,
  watch: false,
  interpreter: "none",
}));

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
      args: `apps/fastapis.ini${process.env.DEVICE != null ? " --device " + process.env.DEVICE : ""}`,
      autorestart: true,
      watch: false,
      interpreter: process.env.PYTHON_BIN
    },
    ...webApps,
  ],
};
