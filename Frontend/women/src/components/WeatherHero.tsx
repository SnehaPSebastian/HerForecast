import WeatherIcon from './WeatherIcon';

interface WeatherHeroProps {
    state?: 'clearing' | 'calm' | 'cloudy' | 'breezy' | 'recovery';
    title?: string;
    message?: string;
}

const stateDefaults: Record<string, { title: string; message: string }> = {
    clearing: {
        title: 'Gentle Clearing',
        message: 'The skies are softly clearing. This is a lovely moment to do something kind for yourself.',
    },
    calm: {
        title: 'Soft Sunshine',
        message: 'Everything feels warm and steady right now. You are beautifully grounded.',
    },
    cloudy: {
        title: 'Quiet Overcast',
        message: 'A little cloudiness is perfectly natural. Be gentle with yourself — this will pass.',
    },
    breezy: {
        title: 'Light Winds',
        message: 'There\'s a little wind in your system. A slow breath might help things settle.',
    },
    recovery: {
        title: 'Rainbow Emerging',
        message: 'Your body is softly recalibrating. This looks like a supportive moment.',
    },
};

export default function WeatherHero({ state = 'clearing', title, message }: WeatherHeroProps) {
    const defaults = stateDefaults[state];

    return (
        <section style={{
            textAlign: 'center',
            padding: 'var(--space-3xl) var(--space-lg) var(--space-xl)',
            animation: 'fade-in-up 1s ease-out',
        }}>
            <WeatherIcon state={state} />

            <p style={{
                fontFamily: 'var(--font-body)',
                fontSize: '0.75rem',
                fontWeight: 500,
                letterSpacing: '0.15em',
                textTransform: 'uppercase',
                color: 'var(--text-light)',
                marginTop: 'var(--space-lg)',
                marginBottom: 'var(--space-sm)',
            }}>
                Your Inner Weather Today
            </p>

            <h1 style={{
                fontFamily: 'var(--font-heading)',
                fontSize: '2rem',
                fontWeight: 500,
                color: 'var(--text-primary)',
                marginBottom: 'var(--space-md)',
                fontStyle: 'italic',
            }}>
                {title || defaults.title}
            </h1>

            <p style={{
                fontFamily: 'var(--font-body)',
                fontSize: '0.95rem',
                fontWeight: 300,
                color: 'var(--text-secondary)',
                lineHeight: 1.7,
                maxWidth: '480px',
                margin: '0 auto',
            }}>
                {message || defaults.message}
            </p>
        </section>
    );
}
