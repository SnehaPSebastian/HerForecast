import InsightCard from './InsightCard';

const NervousSystemIcon = () => (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#D8C7FF" strokeWidth="2" strokeLinecap="round">
        <path d="M12 3C12 3 7 8 7 12C7 16 12 21 12 21C12 21 17 16 17 12C17 8 12 3 12 3Z" />
        <path d="M12 8V16" />
        <path d="M9 11H15" />
    </svg>
);

const EmotionalLoadIcon = () => (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#F8C8DC" strokeWidth="2" strokeLinecap="round">
        <circle cx="12" cy="10" r="7" />
        <path d="M8 21C8 18 10 16 12 16C14 16 16 18 16 21" />
        <path d="M12 7V10" />
        <path d="M10 13Q12 15 14 13" />
    </svg>
);

const RecoveryIcon = () => (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#BEE3F8" strokeWidth="2" strokeLinecap="round">
        <path d="M3 17Q8 7 12 12Q16 17 21 7" />
        <circle cx="12" cy="12" r="2" fill="#BEE3F8" opacity="0.5" />
    </svg>
);

interface InsightsSectionProps {
    nervousSystem?: string;
    emotionalLoad?: string;
    recovery?: string;
}

export default function InsightsSection({
    nervousSystem = 'Balanced and steady',
    emotionalLoad = 'A little cloudy',
    recovery = 'Next gentle pause: 20–40 min',
}: InsightsSectionProps) {
    return (
        <section style={{
            padding: '0 var(--space-lg)',
            marginBottom: 'var(--space-xl)',
        }}>
            <div className="insights-grid" style={{
                display: 'flex',
                gap: 'var(--space-md)',
                flexWrap: 'wrap',
            }}>
                <InsightCard
                    icon={<NervousSystemIcon />}
                    label="Nervous System"
                    status={nervousSystem}
                    accentColor="var(--lavender)"
                    delay={0.1}
                />
                <InsightCard
                    icon={<EmotionalLoadIcon />}
                    label="Emotional Load"
                    status={emotionalLoad}
                    accentColor="var(--blush)"
                    delay={0.25}
                />
                <InsightCard
                    icon={<RecoveryIcon />}
                    label="Recovery"
                    status={recovery}
                    accentColor="var(--sky-blue)"
                    delay={0.4}
                />
            </div>
        </section>
    );
}
