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
