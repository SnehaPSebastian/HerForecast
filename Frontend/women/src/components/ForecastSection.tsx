import { type CSSProperties } from 'react';

interface ForecastSectionProps {
    title?: string;
    message?: string;
}

export default function ForecastSection({
    title = 'Next Hour Outlook',
    message = 'Clouds may gather later this afternoon. A small pause now could keep things lighter.',
}: ForecastSectionProps) {
    const sectionStyle: CSSProperties = {
        padding: '0 var(--space-lg)',
        marginBottom: 'var(--space-xl)',
        animation: 'fade-in-up 1s ease-out 0.5s both',
    };

    const cardStyle: CSSProperties = {
        padding: 'var(--space-xl) var(--space-lg)',
        overflow: 'hidden',
        position: 'relative',
    };

    const labelStyle: CSSProperties = {
        fontFamily: 'var(--font-body)',
        fontSize: '0.7rem',
        fontWeight: 500,
        letterSpacing: '0.12em',
        textTransform: 'uppercase',
        color: 'var(--text-light)',
        marginBottom: 'var(--space-sm)',
        display: 'block',
    };

    const titleStyle: CSSProperties = {
        fontFamily: 'var(--font-heading)',
        fontSize: '1.25rem',
        fontWeight: 500,
        color: 'var(--text-primary)',
        marginBottom: 'var(--space-lg)',
        fontStyle: 'italic',
    };

    const messageStyle: CSSProperties = {
        fontFamily: 'var(--font-body)',
        fontSize: '0.9rem',
        fontWeight: 300,
        color: 'var(--text-secondary)',
        lineHeight: 1.7,
        position: 'relative',
        zIndex: 1,
    };

    return (
        <section style={sectionStyle}>
            <div className="glass-card" style={cardStyle}>
                <span style={labelStyle}>Looking Ahead</span>
                <h2 style={titleStyle}>{title}</h2>

                {/* Horizon SVG */}
                <div className="forecast-horizon" style={{
                    width: '100%',
                    height: '60px',
                    marginBottom: 'var(--space-lg)',
                    borderRadius: 'var(--radius-sm)',
                    overflow: 'hidden',
                    position: 'relative',
                }}>
                    <svg
                        viewBox="0 0 480 80"
                        preserveAspectRatio="none"
                        style={{ width: '100%', height: '100%' }}
                    >
                        <defs>
                            <linearGradient id="horizonSky" x1="0%" y1="0%" x2="0%" y2="100%">
                                <stop offset="0%" stopColor="#EDE4FF" />
                                <stop offset="50%" stopColor="#F8C8DC" stopOpacity="0.5" />
                                <stop offset="100%" stopColor="#FFD6B0" stopOpacity="0.3" />
                            </linearGradient>
                            <linearGradient id="horizonLand" x1="0%" y1="0%" x2="0%" y2="100%">
                                <stop offset="0%" stopColor="#E8DFF5" stopOpacity="0.6" />
                                <stop offset="100%" stopColor="#D8C7FF" stopOpacity="0.2" />
                            </linearGradient>
                        </defs>
                        {/* Sky */}
                        <rect x="0" y="0" width="480" height="80" fill="url(#horizonSky)" />
                        {/* Rolling hills */}
                        <path
                            d="M0,55 Q80,30 160,50 T320,45 T480,55 V80 H0 Z"
                            fill="url(#horizonLand)"
                            style={{ animation: 'horizon-glow 6s ease-in-out infinite' }}
                        />
                        <path
                            d="M0,60 Q120,40 240,58 T480,52 V80 H0 Z"
                            fill="url(#horizonLand)"
                            opacity="0.5"
                        />
                        {/* Sun on horizon */}
                        <circle cx="380" cy="35" r="14" fill="#FFD6B0" opacity="0.6">
                            <animate attributeName="opacity" values="0.4;0.7;0.4" dur="5s" repeatCount="indefinite" />
                        </circle>
                    </svg>
                </div>

                <p style={messageStyle}>
                    {message}
                </p>
            </div>
        </section>
    );
}
