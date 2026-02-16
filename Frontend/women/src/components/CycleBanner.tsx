import { type CSSProperties } from 'react';
import { type CyclePhaseData, cyclePhaseOrder, type CyclePhase } from '../data/cycleData';

interface CycleBannerProps {
    phaseData: CyclePhaseData;
    currentPhase: CyclePhase;
    onPhaseChange: (phase: CyclePhase) => void;
}

const phaseIcons: Record<CyclePhase, React.ReactNode> = {
    menstrual: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
            <path d="M12 2C12 2 6 10 6 15C6 18.3 8.7 21 12 21C15.3 21 18 18.3 18 15C18 10 12 2 12 2Z" />
        </svg>
    ),
    follicular: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
            <path d="M12 22V12M12 12C9 12 6 9 8 5C10 8 12 8 12 8C12 8 14 8 16 5C18 9 15 12 12 12Z" />
            <path d="M8 18Q10 16 12 18Q14 16 16 18" />
        </svg>
    ),
    ovulatory: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
            <circle cx="12" cy="12" r="5" />
            {[0, 45, 90, 135, 180, 225, 270, 315].map((a) => (
                <line key={a} x1="12" y1="3" x2="12" y2="1" transform={`rotate(${a} 12 12)`} />
            ))}
        </svg>
    ),
    luteal: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
            <path d="M18 6C10 6 6 12 6 18" />
            <path d="M12 4C12 4 9 8 7 10" />
            <path d="M20 10C20 10 16 11 14 12" />
            <path d="M16 16C16 16 14 14 12 14" />
        </svg>
    ),
};

const phaseColors: Record<CyclePhase, string> = {
    menstrual: '#D8C7FF',
    follicular: '#BEE3F8',
    ovulatory: '#FFD6B0',
    luteal: '#F8C8DC',
};

export default function CycleBanner({ phaseData, currentPhase, onPhaseChange }: CycleBannerProps) {
    const dotStyle = (phase: CyclePhase): CSSProperties => ({
        width: currentPhase === phase ? '32px' : '10px',
        height: '10px',
        borderRadius: 'var(--radius-full)',
        background: currentPhase === phase
            ? `linear-gradient(90deg, ${phaseColors[phase]}, ${phaseColors[phase]}90)`
            : `${phaseColors[phase]}50`,
        border: 'none',
        cursor: 'pointer',
        transition: 'all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)',
        padding: 0,
    });

    return (
        <section style={{
            padding: '0 var(--space-lg)',
            marginBottom: 'var(--space-xl)',
            animation: 'fade-in-up 0.8s ease-out 0.2s both',
        }}>
            {/* Phase selector dots */}
            <div style={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                gap: '8px',
                marginBottom: 'var(--space-lg)',
            }}>
                {cyclePhaseOrder.map((phase) => (
                    <button
                        key={phase}
                        onClick={() => onPhaseChange(phase)}
                        style={dotStyle(phase)}
                        aria-label={phase}
                    />
                ))}
            </div>

            {/* Phase card */}
            <div className="glass-card" style={{
                padding: 'var(--space-xl)',
                position: 'relative',
                overflow: 'hidden',
            }}>
                {/* Phase icon + label */}
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 'var(--space-sm)',
                    marginBottom: 'var(--space-xs)',
                }}>
                    <div style={{
                        color: phaseColors[currentPhase],
                        display: 'flex',
                        alignItems: 'center',
                        transition: 'color 0.6s ease',
                    }}>
                        {phaseIcons[currentPhase]}
                    </div>
                    <span style={{
                        fontFamily: 'var(--font-body)',
                        fontSize: '0.7rem',
                        fontWeight: 500,
                        letterSpacing: '0.12em',
                        textTransform: 'uppercase',
                        color: 'var(--text-light)',
                    }}>
                        {phaseData.label}
                    </span>
                    <span style={{
                        fontFamily: 'var(--font-body)',
                        fontSize: '0.65rem',
                        color: 'var(--text-light)',
                        fontWeight: 400,
                        opacity: 0.7,
                    }}>
                        {phaseData.dayRange}
                    </span>
                </div>

                {/* Gentle note */}
                <p style={{
                    fontFamily: 'var(--font-body)',
                    fontSize: '0.88rem',
                    fontWeight: 300,
                    color: 'var(--text-secondary)',
                    lineHeight: 1.7,
                    marginBottom: 'var(--space-lg)',
                }}>
                    {phaseData.gentleNote}
                </p>

                {/* Next phase preview */}
                <div style={{
                    background: 'rgba(255, 255, 255, 0.3)',
                    borderRadius: 'var(--radius-md)',
                    padding: 'var(--space-md) var(--space-lg)',
                }}>
                    <span style={{
                        fontFamily: 'var(--font-body)',
                        fontSize: '0.65rem',
                        fontWeight: 500,
                        letterSpacing: '0.1em',
                        textTransform: 'uppercase',
                        color: 'var(--text-light)',
                        display: 'block',
                        marginBottom: '4px',
                    }}>
                        Gently Ahead
                    </span>
                    <p style={{
                        fontFamily: 'var(--font-body)',
                        fontSize: '0.82rem',
                        fontWeight: 300,
                        color: 'var(--text-secondary)',
                        lineHeight: 1.6,
                        fontStyle: 'italic',
                    }}>
                        {phaseData.nextPhasePrep}
                    </p>
                </div>
            </div>
        </section>
    );
}
