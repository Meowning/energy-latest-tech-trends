import enum

class StatusEnum(str, enum.Enum):
    PENDING = "pending"
    PARTIAL = "partial"
    DONE    = "done"

class SourceEnum(str, enum.Enum):
    KOFONS = "한국원자력안전재단"
    KAIF = "한국원자력산업협회"
    