import { type CSSProperties } from 'react';

interface InsightCardProps {
    icon: React.ReactNode;
    label: string;
    status: string;
    accentColor?: string;
    delay?: number;
}

export default function InsightCard({ icon, label, status, accentColor = 'var(--lavender)', delay = 0 }: InsightCardProps) {
    const cardStyle: CSSProperties = {
        padding: 'var(--space-lg)',
        display: 'flex',
        flexDirection: 'column',
        gap: 'var(--space-sm)',
        minWidth: 0,
        flex: 1,
        animation: `fade-in-up 0.8s ease-out ${delay}s both`,
    };

    const iconContainerStyle: CSSProperties = {
        width: '52px',
        height: '52px',
        borderRadius: 'var(--radius-md)',
        background: `linear-gradient(135deg, ${accentColor}40, ${accentColor}20)`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 'var(--space-xs)',
        margin: '0 auto var(--space-xs) auto',
    };

    const labelStyle: CSSProperties = {
        fontFamily: 'var(--font-body)',
        fontSize: '0.78rem',
        fontWeight: 500,
        letterSpacing: '0.08em',
        textTransform: 'uppercase',
        color: 'var(--text-light)',
    };

    const statusStyle: CSSProperties = {
        fontFamily: 'var(--font-heading)',
        fontSize: '1.05rem',
        fontWeight: 500,
        color: 'var(--text-primary)',
        fontStyle: 'italic',
        lineHeight: 1.4,
    };

    return (
        <div className="glass-card" style={cardStyle}>
            <div style={iconContainerStyle}>
                {icon}
            </div>
            <span style={labelStyle}>{label}</span>
            <span style={statusStyle}>{status}</span>
        </div>
    );
}
