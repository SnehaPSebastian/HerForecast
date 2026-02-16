export type CyclePhase = 'menstrual' | 'follicular' | 'fertility' | 'luteal';
export type WeatherState = 'clearing' | 'calm' | 'cloudy' | 'breezy' | 'recovery';

export interface CyclePhaseData {
    phase: CyclePhase;
    label: string;
    weatherState: WeatherState;
    dayRange: string;
    heroTitle: string;
    heroMessage: string;
    nervousSystem: string;
    emotionalLoad: string;
    recovery: string;
    forecastTitle: string;
    forecastMessage: string;
    gentleNote: string;
    nextPhase: CyclePhase;
    nextPhasePrep: string;
}

export const cyclePhases: Record<CyclePhase, CyclePhaseData> = {
    menstrual: {
        phase: 'menstrual',
        label: 'Inner Winter',
        weatherState: 'cloudy',
        dayRange: 'Days 1–5',
        heroTitle: 'Quiet Snowfall',
        heroMessage: 'Your body is in a season of deep rest. Everything is softly slowing down — and that is exactly as it should be.',
        nervousSystem: 'Asking for stillness',
        emotionalLoad: 'Tender and inward',
        recovery: 'Rest is the remedy now',
        forecastTitle: 'The coming days',
        forecastMessage: 'Energy will begin to stir gently over the next few days. For now, rest is your most powerful act.',
        gentleNote: 'Your body is releasing and renewing. Warm drinks, slower mornings, and permission to do less can feel like medicine right now.',
        nextPhase: 'follicular',
        nextPhasePrep: 'A gentle spring is approaching. You may start feeling small sparks of curiosity and lightness in a few days.',
    },
    follicular: {
        phase: 'follicular',
        label: 'Inner Spring',
        weatherState: 'clearing',
        dayRange: 'Days 6–12',
        heroTitle: 'Gentle Clearing',
        heroMessage: 'New energy is stirring like the first warmth of spring reaching through.',
        nervousSystem: 'Rising and steady',
        emotionalLoad: 'Bright and curious',
        recovery: 'Energy building naturally',
        forecastTitle: 'What’s unfolding',
        forecastMessage: 'Creativity and clarity are quietly blooming. This is a beautiful window to start something new or revisit an idea.',
        gentleNote: 'Your body is building fresh energy. This is a lovely time for gentle movement, planning, and saying yes to things that excite you.',
        nextPhase: 'fertility',
        nextPhasePrep: 'A warm, bright phase is on its way. You may soon feel more social, expressive, and radiant.',
    },
    fertility: {
        phase: 'fertility',
        label: 'Inner Summer',
        weatherState: 'calm',
        dayRange: 'Days 13–16',
        heroTitle: 'Full Sunshine',
        heroMessage: 'You are in full bloom. Everything feels warm, open, and beautifully alive. Enjoy this golden light.',
        nervousSystem: 'Vibrant and connected',
        emotionalLoad: 'Warm and open',
        recovery: 'Naturally replenishing',
        forecastTitle: 'Riding this warmth',
        forecastMessage: 'This is your brightest window. Connection, expression, and movement feel effortless right now.',
        gentleNote: 'Your body is at peak vitality. This is a wonderful time for meaningful conversations, creative expression, and being seen.',
        nextPhase: 'luteal',
        nextPhasePrep: 'A quieter, reflective autumn will follow. Begin noticing when you need moments of solitude — it\u2019s your wisdom arriving early.',
    },
    luteal: {
        phase: 'luteal',
        label: 'Inner Autumn',
        weatherState: 'breezy',
        dayRange: 'Days 17–28',
        heroTitle: 'Light Winds Rising',
        heroMessage: 'There\u2019s a gentle wind moving through. Your body is preparing to slow — honour this shift with kindness.',
        nervousSystem: 'Sensitive and watchful',
        emotionalLoad: 'A little wind inside',
        recovery: 'Gentler pacing needed',
        forecastTitle: 'Looking inward',
        forecastMessage: 'You may feel more inward over the coming days. Creating quiet pockets in your schedule can feel deeply nourishing.',
        gentleNote: 'Your body is naturally winding down. Cravings, sensitivity, and the need for comfort are not weaknesses — they are signals of wisdom.',
        nextPhase: 'menstrual',
        nextPhasePrep: 'A deep rest is approaching. Begin softening your plans and creating space for stillness. Your body will thank you.',
    },
};

export const cyclePhaseOrder: CyclePhase[] = ['menstrual', 'follicular', 'fertility', 'luteal'];
