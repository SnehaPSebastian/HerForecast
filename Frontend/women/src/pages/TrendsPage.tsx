import { type CSSProperties } from 'react';

const pageStyle: CSSProperties = {
    padding: 'var(--space-3xl) var(--space-lg) var(--space-xl)',
    animation: 'fade-in-up 0.8s ease-out',
};

const headerStyle: CSSProperties = {
    fontFamily: 'var(--font-heading)',
    fontSize: '1.8rem',
    fontWeight: 500,
    color: 'var(--text-primary)',
    fontStyle: 'italic',
    marginBottom: 'var(--space-sm)',
};

const subtitleStyle: CSSProperties = {
    fontFamily: 'var(--font-body)',
    fontSize: '0.85rem',
    color: 'var(--text-secondary)',
    fontWeight: 300,
    marginBottom: 'var(--space-2xl)',
    lineHeight: 1.6,
};

const weekLabels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const weekData = [
    { label: 'Soft', color: '#D8C7FF', height: 40 },
    { label: 'Steady', color: '#BEE3F8', height: 65 },
    { label: 'Rising', color: '#FFD6B0', height: 55 },
    { label: 'Balanced', color: '#BEE3F8', height: 70 },
    { label: 'Cloudy', color: '#F8C8DC', height: 35 },
    { label: 'Clearing', color: '#D8C7FF', height: 50 },
    { label: 'Steady', color: '#BEE3F8', height: 60 },
];

const trendCards = [
    {
        title: 'Overall Pattern',
        body: 'Your inner weather has been mostly steady this week, with a gentle dip mid-week. That\u2019s completely natural.',
        accent: 'var(--lavender)',
    },
    {
        title: 'Sleep Quality',
        body: 'Your nights have felt restful lately. Your body has been recovering well during sleep.',
        accent: 'var(--sky-blue)',
    },
    {
        title: 'Cycle Awareness',
        body: 'You\u2019re approaching a quieter phase. Gentle pacing this weekend could feel supportive.',
        accent: 'var(--blush)',
    },
];

export default function TrendsPage() {
    return (
        <div style={pageStyle}>
            <p style={{
                fontFamily: 'var(--font-body)',
                fontSize: '0.7rem',
                fontWeight: 500,
                letterSpacing: '0.15em',
                textTransform: 'uppercase',
                color: 'var(--text-light)',
                marginBottom: 'var(--space-sm)',
            }}>
                Your Week
            </p>
            <h1 style={headerStyle}>Patterns & Rhythms</h1>
            <p style={subtitleStyle}>
                A gentle look at how your inner weather has moved over the past seven days.
            </p>

            {/* Weekly bar chart */}
            <div className="glass-card" style={{
                padding: 'var(--space-xl)',
                marginBottom: 'var(--space-xl)',
            }}>
                <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'flex-end',
                    height: '120px',
                    gap: 'var(--space-sm)',
                }}>
                    {weekData.map((d, i) => (
                        <div key={i} style={{
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            gap: '6px',
                            flex: 1,
                        }}>
                            <div style={{
                                width: '100%',
                                maxWidth: '36px',
                                height: `${d.height}%`,
                                borderRadius: 'var(--radius-sm)',
                                background: `linear-gradient(180deg, ${d.color}, ${d.color}60)`,
                                transition: 'height 0.6s ease',
                                minHeight: '16px',
                            }} />
                            <span style={{
                                fontSize: '0.65rem',
                                color: 'var(--text-light)',
                                fontWeight: 500,
                                letterSpacing: '0.03em',
                            }}>
                                {weekLabels[i]}
                            </span>
                        </div>
                    ))}
                </div>

                {/* Legend */}
                <div style={{
                    display: 'flex',
                    justifyContent: 'center',
                    gap: 'var(--space-lg)',
                    marginTop: 'var(--space-lg)',
                    flexWrap: 'wrap',
                }}>
                    {[
                        { color: '#BEE3F8', label: 'Steady' },
                        { color: '#D8C7FF', label: 'Soft' },
                        { color: '#F8C8DC', label: 'Cloudy' },
                        { color: '#FFD6B0', label: 'Rising' },
                    ].map((item, i) => (
                        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                            <div style={{
                                width: '8px', height: '8px', borderRadius: '50%',
                                background: item.color,
                            }} />
                            <span style={{
                                fontSize: '0.7rem', color: 'var(--text-secondary)', fontWeight: 400,
                            }}>
                                {item.label}
                            </span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Trend insight cards */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
                {trendCards.map((card, i) => (
                    <div key={i} className="glass-card" style={{
                        padding: 'var(--space-lg)',
                        animation: `fade-in-up 0.8s ease-out ${0.1 + i * 0.15}s both`,
                    }}>
                        <div style={{
                            width: '6px',
                            height: '6px',
                            borderRadius: '50%',
                            background: card.accent,
                            marginBottom: 'var(--space-sm)',
                        }} />
                        <h3 style={{
                            fontFamily: 'var(--font-heading)',
                            fontSize: '1.05rem',
                            fontWeight: 500,
                            fontStyle: 'italic',
                            color: 'var(--text-primary)',
                            marginBottom: 'var(--space-sm)',
                        }}>
                            {card.title}
                        </h3>
                        <p style={{
                            fontFamily: 'var(--font-body)',
                            fontSize: '0.88rem',
                            fontWeight: 300,
                            color: 'var(--text-secondary)',
                            lineHeight: 1.7,
                        }}>
                            {card.body}
                        </p>
                    </div>
                ))}
            </div>
        </div>
    );
}
