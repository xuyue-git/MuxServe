# 两仓库目录与安装

主仓库保留 MuxServe 调度、FlexStore、flexserver 集成、网关和 xbench。
旁边的 `MuxServe-SGLang-Engine` 对应原 `MuxServe-vLLM` 的引擎角色，仅有六个业务目录：

```text
benchmarks/   扁平 Python 基准入口；调用原生 SGLang CLI，参数不二次封装
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

目标环境：Linux、Python3.10+、四卡 A100、CUDA13.2、Torch2.13.0，另需 Rust 构建工具链。
先准备主仓库 requirements 要求的环境，再执行：

```bash
python tools/bootstrap_source.py
python -m pip install --config-settings editable_mode=compat -e ../MuxServe-SGLang-Engine
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation --config-settings editable_mode=compat -e .
python -m muxserve.preflight configs/sglang_a100.toml
python -m muxserve.launch configs/sglang_a100.toml
```

引擎显式声明 0.5.20，不再靠尚不存在的 Git tag 推导版本。
两个包的 compat editable 模式避免自定义 finder 被严格 preflight 拒绝；
主包关闭构建隔离，使 setup.py 能读取已安装的 Torch。
配置入口现在仅接受实际权重 owner 支持的 TP4，提早拒绝 TP1/TP2。
模型路径、revision/tokenizer 和个人 asset_lock_dir 仍需按实际机器填写。

## 依赖边界与源码锁

主仓库 import 已安装的 `sglang`；已有引擎 hooks 仍调用主仓库集成模块，协议不变。
`upstream-reference/` 保留上游 commit、归档摘要、原13补丁和清理后的保留文件映射。
`python tools/bootstrap_source.py` 验证原始来源、保留文件及已审阅修改；
安装后的 preflight 继续检查原13补丁，并补充检查本次修改的包内文件。
引擎不再带 `muxserve-integration`、额外审计记录、补丁管理器或完整上游文档站点。
历史补丁管理入口已经退役；变更引擎源码后需要在评审中更新兼容性锁。

本地引擎 Git 仍无 authored commit，也未导入上游历史或推送远端。
用户已选择之后自行实际验证，所以本次仅做结构、语法、源码身份与本地回归检查，
没有安装整套引擎、启动模型、执行 GPU/ABI 或性能验证。

## 回滚

旧引擎完整目录、主仓库改动前记录及逐文件来源保存在主项目外：
`.codex-artifacts/engine-parity-audit-20261114/`。
恢复前先比较后续编辑，再还原主仓库相关文件和旧引擎目录。xbench 与个人 IDE 文件保留。
