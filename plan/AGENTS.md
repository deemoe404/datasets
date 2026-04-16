# plan

本文件适用于 `./plan` 目录及其子目录，主要记录这里的 LaTeX 构建约定，避免 agent 因找不到 TeX 工具而误判环境。

## LaTeX 环境

- 不要假设系统 TeX 在 `/usr/local`、`/usr/bin` 或需要 `sudo`。
- 本机可用的 TeX Live 已安装在可写用户目录：
  - `/home/sam/.local/texlive/2026`
- TeX 可执行文件目录：
  - `/home/sam/.local/texlive/2026/bin/x86_64-linux`
- 已验证可用的命令：
  - `pdflatex`
  - `xelatex`
  - `lualatex`
  - `tlmgr`
- 版本为 `TeX Live 2026`。

## Shell 与 PATH

- 路径配置已经写入：
  - `/home/sam/.profile`
  - `/home/sam/.zshrc`
- 新开的 login shell 或 zsh 通常可以直接找到 LaTeX。
- 如果当前终端或 agent 进程找不到 `pdflatex`，先执行：

```sh
source ~/.zshrc
```

- 如果仍需避免 shell 启动文件差异，直接使用绝对路径调用：

```sh
/home/sam/.local/texlive/2026/bin/x86_64-linux/pdflatex
/home/sam/.local/texlive/2026/bin/x86_64-linux/xelatex
/home/sam/.local/texlive/2026/bin/x86_64-linux/lualatex
/home/sam/.local/texlive/2026/bin/x86_64-linux/tlmgr
```

## 本目录构建约定

- 本目录的 LaTeX 临时输出目录使用 `./latex-tmp`。
- 仓库里已有可复用脚本：
  - `../scripts/latex_build_local.sh`
- 该脚本已经固定使用：
  - `/home/sam/.local/texlive/2026/bin/x86_64-linux/pdflatex`

示例：

```sh
bash ../scripts/latex_build_local.sh /home/sam/datasets/plan/wind_farm_design_revised_v3_input_reorg.tex
```

## 已知状态

- `pdflatex` / `xelatex` / `lualatex` / `tlmgr` 均已验证可执行。
- `pdflatex` 已成功编译过最小 `.tex` 文件并生成 PDF。
- `tlmgr` 当前安装根为 `/home/sam/.local/texlive/2026`。
- 安装目录大小约 `9.6G`。
- `hausarbeit-jura` 在安装时首次校验失败，重试后成功；`tlmgr info hausarbeit-jura` 显示 `installed: Yes`。

## 维护

- 更新 TeX Live 时使用：

```sh
tlmgr update --self --all
```

- 如果 agent 报告 “找不到 latex / pdflatex / tlmgr”，优先检查 PATH 是否已加载；不要重复尝试系统包管理器安装。
