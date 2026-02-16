from typing import Dict

# Mapping of menstrual phases to corresponding mood states
PHASE_TO_MOOD: Dict[str, str] = {
    "Menstrual": "Rest & Restore",
    "Follicular": "Light & Energized",
    "Fertility": "Magnetic & Expressive",
    "Luteal": "Reflective & Deep"
}

def get_mood_from_phase(phase: str) -> str:
    """
    Get the mood state corresponding to the predicted menstrual phase.
    Raises ValueError if phase is not recognized.
    """
    if phase not in PHASE_TO_MOOD:
        raise ValueError(f"Unknown phase: {phase}")
    return PHASE_TO_MOOD[phase]
