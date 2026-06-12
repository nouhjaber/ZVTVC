from .whisper_teacher import (
    WhisperTeacher,
    create_whisper_teacher,
    WHISPER_MODELS,
    get_whisper_model_name
)

from .mhubert_teacher import (
    MHubertTeacher,
    create_mhubert_teacher,
    HUBERT_MODELS,
    get_hubert_model_name
)

from .ema_teacher import (
    EMATeacher,
    EMAScheduler,
    create_ema_teacher,
    get_recommended_alpha,
    EMA_ALPHA_VALUES
)

from .teacher_manager import (
    TeacherManager,
    create_teacher_manager
)

__all__ = [
    # Teacher classes
    'WhisperTeacher',
    'MHubertTeacher',
    'EMATeacher',
    'EMAScheduler',
    'TeacherManager',

    # Factory functions
    'create_whisper_teacher',
    'create_mhubert_teacher',
    'create_ema_teacher',
    'create_teacher_manager',

    # Constants and utilities
    'WHISPER_MODELS',
    'HUBERT_MODELS',
    'EMA_ALPHA_VALUES',
    'get_whisper_model_name',
    'get_hubert_model_name',
    'get_recommended_alpha',
]

# Version info
__version__ = '3.2.0'
__author__ = 'Content Encoder Team'
