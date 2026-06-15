"""
HiLoRM 端点加载器 (共享核心)。

读取统一配置 config.yaml (路径取自 $HILORM_CONFIG, 默认为本仓库根目录的 config.yaml),
按 $HILORM_STAGE 取出当前 stage 的端点, 暴露为模块级对象 ``EP``。

各 stage 的 ``agentless/util/api_requests.py`` 与共享的 ``agentless/fl/Index.py`` 这样用::

    from agentless.util.endpoints import EP
    ...
    sgl.RuntimeEndpoint(EP.sglang_rm)
    openai.OpenAI(base_url=EP.generation_base_url, api_key=EP.generation_api_key)
    OpenAI(base_url=EP.judge_base_url, api_key=EP.judge_api_key)  # model=EP.judge_model

设计要点:
- 仅依赖 ``os`` + ``yaml``, 不 import 任何 agentless 模块 -> 无循环依赖。
- import 阶段即加载, 但 stage 未设置 / 端点缺失时返回 None 而不报错 -> 不阻塞 import 冒烟测试;
  真正缺失的端点只在运行期被使用时由下游服务报错。
- 配置文件本身缺失才抛错 (FileNotFoundError), 给出清晰提示。
"""

import os

import yaml


def _find_config_path():
    """优先 $HILORM_CONFIG; 否则取本文件相对的仓库根 config.yaml。

    本文件位于 ``<root>/core/agentless/util/endpoints.py`` ->  ../../.. == <root>。
    """
    env = os.environ.get("HILORM_CONFIG")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    return os.path.join(root, "config.yaml")


class _Endpoints:
    """当前 stage 的端点视图。缺失项一律为 None。"""

    def __init__(self, cfg, stage):
        self.config_path = cfg.get("__path__")
        self.stage = stage

        common = cfg.get("common", {}) or {}
        stages = cfg.get("stages", {}) or {}
        st = stages.get(stage, {}) or {}
        paths = cfg.get("paths", {}) or {}

        # --- 生成 (所有 stage 相同) ---
        gen = common.get("generation", {}) or {}
        self.generation_base_url = gen.get("base_url")
        self.generation_api_key = gen.get("api_key", "empty")

        # --- embedding (仅 step3) ---
        emb = common.get("embedding", {}) or {}
        self.embedding_model_name = emb.get("model_name", "text-embedding-3-small")
        self.embedding_api_base = emb.get("api_base")
        self.embedding_api_key = emb.get("api_key")

        # --- 生成式 RM (SGLang) ---
        self.sglang_rm = st.get("sglang_rm")

        # --- LLM-as-judge ---
        judge = st.get("judge") or {}
        self.judge_base_url = judge.get("base_url")
        self.judge_api_key = judge.get("api_key", "empty")
        self.judge_model = judge.get("model", "YiRM")

        # --- 标量 RM (Skywork, 仅 step6) ---
        sky = st.get("skywork") or {}
        self.skywork_url = sky.get("url")
        self.skywork_model = sky.get("model", "sk8b")
        self.skywork_model_path = sky.get("model_path")

        # --- 外部数据路径 ---
        self.project_file_loc = paths.get("project_file_loc")
        self.pred_list_root = paths.get("pred_list_root")
        self.start_file_root = paths.get("start_file_root")
        self.embedding_persist_dir = paths.get("embedding_persist_dir")

    def require(self, attr):
        """取一个端点, 若为 None 则抛出清晰错误 (供运行期主动校验, 可选使用)。"""
        val = getattr(self, attr, None)
        if val is None:
            raise RuntimeError(
                f"HiLoRM config: '{attr}' 未配置 (stage={self.stage!r}, "
                f"config={self.config_path!r})。请检查 config.yaml。"
            )
        return val

    def __repr__(self):
        return (
            f"<HiLoRM EP stage={self.stage!r} gen={self.generation_base_url!r} "
            f"sglang_rm={self.sglang_rm!r} judge={self.judge_base_url!r} "
            f"skywork={self.skywork_url!r}>"
        )


def load(stage=None, config_path=None):
    """加载配置并返回当前 stage 的 _Endpoints。stage 默认取 $HILORM_STAGE。"""
    stage = stage if stage is not None else os.environ.get("HILORM_STAGE")
    path = config_path or _find_config_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"HiLoRM config 未找到: {path} 。请设置 $HILORM_CONFIG 或在仓库根放置 config.yaml。"
        )
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["__path__"] = path
    return _Endpoints(cfg, stage)


# 模块级单例: import 时即按 $HILORM_STAGE 加载。
EP = load()
