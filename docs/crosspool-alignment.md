# 与 CrossPool 启动方式的对照

对照对象：CrossPool `3c1f976cdd1069756314ecb7142581fad2367272`（xbench 客户端锁定版本）的
README、`docs/tutorials/quick-start.md`、`docs/tutorials/tooling.md`、`pyproject.toml` 和
`src/xpool-dev/` 源码。下文的 CrossPool 路径都指这个版本。

## 已经一样的

| 方面 | CrossPool | MuxServe |
|---|---|---|
| 代码怎么到机器上 | 仓库 checkout（quick-start：“Configure the checkout ... From the repository root”） | 两个仓库 git clone 并排放 |
| 依赖与环境 | 根 `pyproject.toml` + `uv.lock`，`uv sync` | 相同 |
| Python | uv 管理的 Python 3.12（`requires-python = ">=3.12,<3.13"`，`python-preference = "only-managed"`，`python-downloads = "automatic"`） | 相同 |
| uv 版本 | `required-version = ">=0.12.17"` | 相同 |
| 机器相关设置 | 被 git 忽略的 `.env` 和 `configs/*.local.toml`，从 `*.example` 复制 | 被 git 忽略的 `.env` 和 `configs/muxserve.local.toml`，从 `.env.example`、`configs/muxserve.example.toml` 复制；用哪四张卡写在启动命令里 |
| 每个终端 | `export UV_ENV_FILE="$PWD/.env"` | 相同 |
| 配置文件格式 | TOML | TOML（部署配置、本地设置、消融配置；启动器内部写给子进程的解析结果仍是 JSON） |
| 运行命令形状 | `uv run xpool <子命令>` | `uv run muxserve serve CONFIG.toml` |
| 查看实际生效的配置 | `uv run xpool config dump` | `uv run muxserve config dump CONFIG.toml` |
| 基准用例 | `benches.toml`，严格 schema | 同一 schema（已用 CrossPool 的 `case.py` 校验 61 个用例） |
| 跑基准 | `uv run xbench run --case <前缀>` | 相同：在 CrossPool 的 `configs/xkit.local.toml` 设 `[xbench] catalog = "../../MuxServe/benchmark/xbench/benches.toml"` 后，命令完全一样 |
| 报告 | `uv run xbench report RUN_ID` | 相同 |

最后一行的依据：xbench 的目录路径本身就是一个配置项，`src/xpool-dev/xkit/config.py` 里
`ConfigSetting(name="xbench_catalog", path=("xbench", "catalog"), default="benches/benches.toml", cli="--catalog")`，
`configs/xkit.example.toml` 里写的是 `[xbench] catalog = "../benches/benches.toml"`。

## 不能一样的，以及原因

### 1. 服务由谁启动：CrossPool 用 owned 模式，我们只能用 client 模式

**原理。** owned 用例由 xbench 自己起服务、跑负载、关服务，它能起的只有 CrossPool
自己的部署：xpool 的 ATN/FFN 拓扑加 SGLang 实例。

**代码证据。**
- `src/xpool-dev/xbench/harness/serving/case.py`：
  `class OwnedBenchCase` 有 `mode: Literal["owned"]` 和 `deployment: Path`；
  `resolve_deployment(case)` 把它交给 `assemble_config(...)`，返回类型是 `XpoolConfig`，
  再构造 `ServingLaunch(config, {}, Path.cwd(), tuple(SglangLaunchModel(...)))`。
- `src/xpool-dev/xkit/serving/sglang/launch.py`：`ServingLaunch.config: XpoolConfig`；
  `__post_init__` 要求要起的模型集合和 `config.models` 完全相同，并检查
  `config.atn_tp_size_of(...)`、`atn_dp_size` 这些 xpool 拓扑字段。

MuxServe 的部署（私有 MPS、每模型每卡的权重进程、P/D rank、主控）不是 `XpoolConfig`，
也不是若干个 `sglang serve` 实例。要用 owned 模式，必须在 CrossPool 里新增一种部署类型和
启动器，也就是改 CrossPool 的代码。原版 MuxServe 同理。

**影响。** 服务按 README 的命令手动启动；每个 case、每一轮之间重启服务，报告里写明是 client 模式。

### 2. xbench 不和服务装在同一个环境

**原理。** CrossPool 根 workspace 把 `xpool`（生产运行时）和 `xpool-dev`（xbench）装进同一个环境。
xbench 依赖 `xpool`：

- `src/xpool-dev/pyproject.toml`：`dependencies = ["xpool", ..., "sglang==0.5.20", ...]`；
- 根 `pyproject.toml`：`xpool` 需要 CMake / CUDA / NVSHMEM 原生编译（`nvidia-nvshmem-cu13`、
  `cuda-bindings`、`ortools` 等），并注册 SGLang 插件
  `[project.entry-points."sglang.srt.plugins"] xpool = "xpool.integrations.sglang.plugin:install"`。

SGLang 会加载环境里所有已安装的通用插件，除非 `SGLANG_PLUGINS` 做了限制：
引擎 `sglang/srt/plugins/__init__.py` 第 51–69 行，`SGLANG_PLUGINS` 为空时 `allowed_set` 为 `None`，
`entry_points(group=...)` 找到的插件全部加载。

装进同一个环境，就等于把另一个服务系统的原生扩展和 SGLang 插件放进 MuxServe 的推理进程，
正是评审会要求避免的“同一环境混装不同引擎、底层库冲突”。所以 xbench 留在 CrossPool
自己的环境里。命令形式不受影响，只是在 CrossPool 目录下运行。

### 3. 启动服务的命令条数不同

CrossPool 起一个服务要四个进程、四个终端（quick-start）：`uv run xpool daemon serve`、
`uv run xpool atnagent --device 0`、`uv run xpool ffnagent --device 1`、
`uv run xpool exec -- sglang serve ...`。这是因为它把注意力和 FFN 放在不同的卡上，各有一个代理进程。
MuxServe 只有一个启动器，它自己拉起 MPS、主控、权重进程和 P/D rank，所以只有一条
`uv run muxserve serve CONFIG.toml`。差别来自两个系统的结构，不是写法。

### 4. 部署配置的内容不同

两边都是 TOML，但 CrossPool 的部署文件描述 xpool 拓扑（ATN/FFN 设备、实例几何），
我们的描述 MuxServe 部署（模型、TP4、KV 池、MPS 比例、调度开关）。字段由各自的系统决定，不能共用。

### 5. 原版 MuxServe 的环境没有改成 uv

原版跑在 Python 3.9、torch 2.0.1 + CUDA 11.8、从源码编译的 MuxServe-vLLM 上。
它的 `Muxserve-change` 分支只加了 HTTP 入口（按“原版只做最小修改”的要求）；
改它的安装方式会改变基线本身，所以没有做，沿用服务器上已验证的旧环境。它的配置和完整启动
命令在该分支的 `examples/xbench/`：用原版自己的 `python -m muxserve.launch` 加载
`examples/xbench/fig5-a0.7-s1.0.yaml`，多一个 `--http-port 18080`，启动前先起一个私有 MPS
（原版的启动器不会自己起）。客户端命令和上面完全相同。

## 尚未验证

- 服务端改用 Python 3.12 后还没有在 A100 上跑过；以前的 GPU 结果来自 Python 3.10 的环境。
- 上面的命令没有在服务器上整体走过一遍。
