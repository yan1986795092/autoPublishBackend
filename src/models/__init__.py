from sqlalchemy import create_engine, Column, String, Text, Integer, Float, Boolean, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid
import enum

Base = declarative_base()


def generate_uuid():
    return str(uuid.uuid4())


class ContentStatus(str, enum.Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class PublishStatus(str, enum.Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    nickname = Column(String(50))
    quota_daily = Column(Integer, default=10)
    quota_used_today = Column(Integer, default=0)
    quota_reset_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    contents = relationship("Content", back_populates="user", cascade="all, delete-orphan")
    templates = relationship("PromptTemplate", back_populates="user")
    platform_accounts = relationship("PlatformAccount", back_populates="user")


class Content(Base):
    __tablename__ = "contents"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    title = Column(String(255))
    topic = Column(String(500))
    platform_code = Column(String(50))
    content_md = Column(Text)
    content_html = Column(Text)
    status = Column(SQLEnum(ContentStatus), default=ContentStatus.DRAFT)
    tags = Column(String(500))
    metadata_json = Column(Text)
    view_count = Column(Integer, default=0)
    like_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="contents")
    versions = relationship("ContentVersion", back_populates="content", cascade="all, delete-orphan")
    publish_records = relationship("PublishRecord", back_populates="content")


class ContentVersion(Base):
    __tablename__ = "content_versions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    content_id = Column(String(36), ForeignKey("contents.id"), nullable=False)
    version_number = Column(Integer, nullable=False)
    content_md = Column(Text)
    content_html = Column(Text)
    change_summary = Column(Text)
    created_at = Column(DateTime, default=func.now())

    content = relationship("Content", back_populates="versions")


class PromptTemplate(Base):
    __tablename__ = "prompt_templates"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    platform_code = Column(String(50))
    name = Column(String(100), nullable=False)
    description = Column(Text)
    system_prompt = Column(Text)
    user_template = Column(Text)
    variables_json = Column(Text)
    model_preferred = Column(String(50))
    temperature = Column(Float, default=0.7)
    max_tokens = Column(Integer, default=2000)
    is_public = Column(Boolean, default=False)
    use_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="templates")


class Platform(Base):
    __tablename__ = "platforms"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    code = Column(String(50), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    icon_url = Column(String(255))
    category = Column(String(50), default='social_media')
    is_active = Column(Boolean, default=True)
    is_mock = Column(Boolean, default=True)

    accounts = relationship("PlatformAccount", back_populates="platform")


class PlatformAccount(Base):
    __tablename__ = "platform_accounts"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    platform_id = Column(String(36), ForeignKey("platforms.id"), nullable=False)
    account_name = Column(String(100))
    access_token_encrypted = Column(Text)
    extra_config = Column(Text)
    is_active = Column(Boolean, default=True)
    last_used_at = Column(DateTime)
    created_at = Column(DateTime, default=func.now())

    user = relationship("User", back_populates="platform_accounts")
    platform = relationship("Platform", back_populates="accounts")


class ChatConversation(Base):
    __tablename__ = "chat_conversations"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(255), nullable=False, default="新对话")
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    user = relationship("User", backref="chat_conversations")
    messages = relationship(
        "ChatMessageRecord",
        back_populates="conversation",
        cascade="all, delete-orphan",
    )


class ChatMessageRecord(Base):
    __tablename__ = "chat_message_records"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    conversation_id = Column(
        String(36), ForeignKey("chat_conversations.id"), nullable=False, index=True
    )
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    status = Column(String(20), default="completed")
    result_json = Column(Text)
    task_id = Column(String(36), nullable=True)
    plan_json = Column(Text)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    conversation = relationship("ChatConversation", back_populates="messages")


class PublishRecord(Base):
    __tablename__ = "publish_records"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    content_id = Column(String(36), ForeignKey("contents.id"), nullable=False)
    platform_code = Column(String(50), nullable=False)
    platform_post_id = Column(String(255))
    platform_url = Column(Text)
    status = Column(SQLEnum(PublishStatus), default=PublishStatus.PENDING)
    error_message = Column(Text)
    response_data = Column(Text)
    published_at = Column(DateTime)
    created_at = Column(DateTime, default=func.now())

    content = relationship("Content", back_populates="publish_records")

class Preset(Base):
    """用户预设"""
    __tablename__ = "presets"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    parameters_json = Column(Text, nullable=False)
    platform_code = Column(String(50))
    template_id = Column(String(36))
    persona_id = Column(String(36), ForeignKey("personas.id"), nullable=True)
    use_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    user = relationship("User", backref="presets")


class Persona(Base):
    """人设"""
    __tablename__ = "personas"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    tone = Column(String(50))
    style_notes = Column(Text)
    is_active = Column(Boolean, default=True)
    use_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class AgentTask(Base):
    """Agent 任务表（持久化异步 Agent 任务）"""
    __tablename__ = "agent_tasks"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    conversation_id = Column(String(36), nullable=True, index=True)
    task_type = Column(String(50), default="content_generation")
    status = Column(String(20), default="pending")
    prompt = Column(Text, nullable=False)
    input_summary = Column(String(500))
    execution_plan = Column(Text)
    total_steps = Column(Integer, default=1)
    completed_steps = Column(Integer, default=0)
    progress = Column(Float, default=0.0)
    total_tokens = Column(Integer, default=0)
    total_cost = Column(Float, default=0.0)
    result_json = Column(Text)
    error_message = Column(Text)
    thinking_trace_json = Column(Text)
    agent_events_json = Column(Text)
    created_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime)

    user = relationship("User", backref="agent_tasks")
    subtasks = relationship("SubTask", back_populates="agent_task", cascade="all, delete-orphan")


class SubTask(Base):
    """子任务（Agent 执行步骤记录）"""
    __tablename__ = "agent_subtasks"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    task_id = Column(String(36), ForeignKey("agent_tasks.id"), nullable=False)
    agent_type = Column(String(50), nullable=False)
    status = Column(String(20), default="pending")
    input_json = Column(Text)
    output_json = Column(Text)
    tokens_used = Column(Integer, default=0)
    time_elapsed = Column(Float, default=0.0)
    retry_count = Column(Integer, default=0)
    error_message = Column(Text)
    created_at = Column(DateTime, default=func.now())

    agent_task = relationship("AgentTask", back_populates="subtasks")


class ThinkingTraceRecord(Base):
    """Thinking Trace 持久化记录"""
    __tablename__ = "agent_thinking_traces"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    task_id = Column(String(36), ForeignKey("agent_tasks.id"), nullable=False, index=True)
    subtask_id = Column(String(36), ForeignKey("agent_subtasks.id"), nullable=True)
    step_number = Column(Integer, nullable=False)
    agent = Column(String(50))
    action = Column(String(100))
    reasoning = Column(Text)
    input_summary = Column(Text)
    output_summary = Column(Text)
    tokens_used = Column(Integer, default=0)
    time_elapsed = Column(Float, default=0.0)
    status = Column(String(20), default="running")
    created_at = Column(DateTime, default=func.now())

    # agent_task = relationship("AgentTask")


class ToolCall(Base):
    """工具调用记录（持久化 Agent 的工具使用日志）

    对应 AGENT_ARCHITECTURE.md §11.1 ToolCall 表，
    用于成本审计和性能分析（保留 30 天）。
    """
    __tablename__ = "agent_tool_calls"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    subtask_id = Column(String(36), ForeignKey("agent_subtasks.id"), nullable=True, index=True)
    task_id = Column(String(36), ForeignKey("agent_tasks.id"), nullable=False, index=True)
    tool_name = Column(String(100), nullable=False)
    parameters_json = Column(Text)
    result_json = Column(Text)
    status = Column(String(20), default="running")
    error_message = Column(Text)
    tokens_used = Column(Integer, default=0)
    time_elapsed = Column(Float, default=0.0)
    created_at = Column(DateTime, default=func.now())


class SkillCategory(str, enum.Enum):
    WRITING = "writing"
    ANALYSIS = "analysis"
    CODING = "coding"
    RESEARCH = "research"
    DESIGN = "design"
    SOCIAL = "social"
    CUSTOM = "custom"


class SkillSource(str, enum.Enum):
    BUILTIN = "builtin"
    USER = "user"
    IMPORTED = "imported"


class Skill(Base):
    """用户自定义/导入的创作 Skill

    支持外部导入 SKILL.md 解析后的结构，
    用于在 Agent 执行时注入特定的 system_prompt 和 plan 配置。
    """
    __tablename__ = "skills"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)

    # 元信息
    name = Column(String(100), nullable=False)
    description = Column(Text)
    category = Column(SQLEnum(SkillCategory), default=SkillCategory.WRITING)
    icon = Column(String(10), default="✍️")
    source = Column(SQLEnum(SkillSource), default=SkillSource.USER)
    author = Column(String(100))
    version = Column(String(20))
    tags = Column(String(500))          # 逗号分隔
    is_enabled = Column(Boolean, default=True)
    use_count = Column(Integer, default=0)

    # 执行相关
    system_prompt = Column(Text, nullable=False)
    user_prompt_template = Column(Text)  # 可选：包含 {topic} {content} 占位符的模板
    preset_plan_json = Column(Text)      # JSON: { complexity, agent_sequence }
    content_types_json = Column(Text)    # JSON: ["blog", "social_post"]

    # 配置 schema（用户可在 UI 上覆盖的字段）
    config_schema_json = Column(Text)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    user = relationship("User", backref="skills")

    @property
    def tags_list(self) -> list[str]:
        if not self.tags:
            return []
        return [t.strip() for t in self.tags.split(",") if t.strip()]

    @property
    def preset_plan(self) -> dict | None:
        if not self.preset_plan_json:
            return None
        import json
        try:
            return json.loads(self.preset_plan_json)
        except Exception:
            return None

    @property
    def content_types(self) -> list[str]:
        if not self.content_types_json:
            return []
        import json
        try:
            return json.loads(self.content_types_json)
        except Exception:
            return []

    @property
    def config_schema(self) -> dict | None:
        if not self.config_schema_json:
            return None
        import json
        try:
            return json.loads(self.config_schema_json)
        except Exception:
            return None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description or "",
            "category": self.category.value,
            "icon": self.icon,
            "source": self.source.value,
            "author": self.author,
            "version": self.version,
            "tags": self.tags_list,
            "is_enabled": self.is_enabled,
            "use_count": self.use_count,
            "system_prompt": self.system_prompt,
            "user_prompt_template": self.user_prompt_template,
            "preset_plan": self.preset_plan,
            "content_types": self.content_types,
            "config_schema": self.config_schema,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def to_meta_dict(self) -> dict:
        """列表视图 — 不包含 system_prompt 和 config_schema"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description or "",
            "category": self.category.value,
            "icon": self.icon,
            "source": self.source.value,
            "author": self.author,
            "version": self.version,
            "tags": self.tags_list,
            "is_enabled": self.is_enabled,
            "use_count": self.use_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class SnapshotRecord(Base):
    """执行快照元数据（完整 JSON 存文件系统）"""
    __tablename__ = "execution_snapshots"

    snapshot_id = Column(String(32), primary_key=True)
    task_id = Column(String(36), ForeignKey("agent_tasks.id"), nullable=True, index=True)
    agent_name = Column(String(64), index=True)
    timestamp = Column(DateTime, index=True)
    duration_ms = Column(Integer, default=0)
    tags = Column(Text, nullable=True)                    # JSON list

    template_names = Column(Text, nullable=True)          # JSON list
    template_versions = Column(Text, nullable=True)       # JSON list
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    model = Column(String(64), nullable=True)
    result_summary = Column(Text, nullable=True)
    captured_hash = Column(String(64))
    parent_snapshot_id = Column(String(32), nullable=True)

    snapshot_path = Column(String(256), nullable=False)   # "snapshots/202606/snap_xxx.json"
