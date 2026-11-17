# 两仓库目录与安装

主仓库保留 MuxServe 调度、FlexStore、flexserver 集成、网关和 xbench 用例。
旁边的 `MuxServe-SGLang-Engine` 对应原 `MuxServe-vLLM` 的引擎角色，只有五个业务目录（压测用主仓库的 xbench 用例）：

```text
csrc/         Rust 与 protobuf 构建源
docs/        引擎接口说明
examples/     上游离线、异步和服务代码示例
tests/       语言模型相关的上游单元测试
sglang/       原生 Python 引擎包
```

引擎安装入口是其根目录，包名仍为 `sglang`，`sglang.srt` 下原生文件名不变。
本次未将 SGLang 调度器伪装成 vLLM 的文件结构；只把原 `python/sglang` 包移到根目录。
构建用的 Rust/proto 同时迁入 csrc，并修正两个 JIT lookup 和 setup 路径。

## 安装

两个仓库并排 clone（引擎目录名必须是 `MuxServe-SGLang-Engine`），在主仓库根目录：

```bash
python tools/bootstrap_source.py      # 校验引擎源码与 upstream-reference 的记录一致
uv sync                               # 按 uv.lock 装两个包（引擎 editable）和全部依赖
export UV_ENV_FILE="$PWD/.env"
CUDA_VISIBLE_DEVICES=... uv run muxserve serve CONFIG.toml
```

环境：Linux x86_64、四卡 A100、CUDA Toolkit 13.2（`nvcc`）、uv 0.12.17 以上；Python 3.12 由 uv 管理。
主仓库 `pyproject.toml` 用 `[tool.uv.sources]` 指向 `../MuxServe-SGLang-Engine`，所以只有 uv 这一条安装路径。
机器上没有 Rust（`cargo`）时，`uv sync` 前加 `SGLANG_BUILD_RUST_EXTS=none`：跳过的 Rust 扩展
（gRPC、多模态、Rust HTTP 服务、Rust 前缀树）本项目都不用。完整步骤见 `README.md` 和 `benchmark/xbench/README.md`。

引擎显式声明 0.5.20，不再靠尚不存在的 Git tag 推导版本。
两个包用 compat editable 模式，避免自定义 finder 被严格 preflight 拒绝。
配置入口只接受实际权重 owner 支持的 TP4，提早拒绝 TP1/TP2。
模型路径和个人 `asset_lock_dir` 写在本机设置文件 `configs/muxserve.local.toml` 里。

## 依赖边界与源码锁

主仓库 import 已安装的 `sglang`；引擎里的 hooks 调用主仓库集成模块，由环境变量 `MUXSERVE_EXECUTION_GROUP`
控制，只在 MuxServe 启动的进程里生效。
`upstream-reference/` 保留上游 commit、归档摘要、原 13 个补丁和改动文件的哈希。
`python tools/bootstrap_source.py` 验证原始来源、保留文件及已审阅修改；
启动前的 preflight 再检查已安装包里这些文件的哈希。
变更引擎源码后，需要同时更新 `upstream-reference/engine-repository.json` 里的哈希。

引擎仓库：https://github.com/xuyue-git/Muxserve-SGLang-Engine （`main`）。
