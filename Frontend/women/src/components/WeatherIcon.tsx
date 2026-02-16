import { type CSSProperties } from 'react';

interface WeatherIconProps {
    state?: 'clearing' | 'calm' | 'cloudy' | 'breezy' | 'recovery';
}

export default function WeatherIcon({ state = 'clearing' }: WeatherIconProps) {
    const containerStyle: CSSProperties = {
        width: '200px',
        height: '200px',
        position: 'relative',
        margin: '0 auto',
        animation: 'float 6s ease-in-out infinite',
    };

    const glowStyle: CSSProperties = {
        position: 'absolute',
        top: '50%',
        left: '50%',
        width: '220px',
        height: '220px',
        transform: 'translate(-50%, -50%)',
        borderRadius: '50%',
        background: 'radial-gradient(circle, rgba(255,214,176,0.4) 0%, rgba(248,200,220,0.2) 40%, transparent 70%)',
        animation: 'breathe 4s ease-in-out infinite',
        pointerEvents: 'none',
    };

    const renderIcon = () => {
        switch (state) {
            case 'clearing':
            case 'calm':
                return (
                    <svg viewBox="0 0 200 200" style={{ width: '100%', height: '100%' }}>
                        {/* Sun body */}
                        <defs>
                            <radialGradient id="sunGrad" cx="50%" cy="50%" r="50%">
                                <stop offset="0%" stopColor="#FFE4B5" />
                                <stop offset="60%" stopColor="#FFD6B0" />
                                <stop offset="100%" stopColor="#F8C8DC" />
                            </radialGradient>
                            <filter id="sunGlow">
                                <feGaussianBlur stdDeviation="3" result="blur" />
                                <feMerge>
                                    <feMergeNode in="blur" />
                                    <feMergeNode in="SourceGraphic" />
                                </feMerge>
                            </filter>
                        </defs>
                        {/* Sun rays */}
                        <g style={{ animation: 'gentle-spin 30s linear infinite', transformOrigin: '100px 100px' }}>
                            {[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330].map((angle) => (
                                <line
                                    key={angle}
                                    x1="100"
                                    y1="30"
                                    x2="100"
                                    y2="18"
                                    stroke="#FFD6B0"
                                    strokeWidth="2.5"
                                    strokeLinecap="round"
                                    opacity="0.6"
                                    transform={`rotate(${angle} 100 100)`}
                                />
                            ))}
                        </g>
                        {/* Sun circle */}
                        <circle cx="100" cy="100" r="48" fill="url(#sunGrad)" filter="url(#sunGlow)" />
                        {/* Sun face — gentle smile */}
                        <path
                            d="M82 108 Q100 122 118 108"
                            fill="none"
                            stroke="#D4A574"
                            strokeWidth="2"
                            strokeLinecap="round"
                            opacity="0.5"
                        />
                        {/* Small cloud */}
                        {state === 'clearing' && (
                            <g style={{ animation: 'drift-right 8s ease-in-out infinite' }} opacity="0.6">
                                <ellipse cx="145" cy="135" rx="28" ry="14" fill="white" />
                                <ellipse cx="135" cy="128" rx="18" ry="12" fill="white" />
                                <ellipse cx="155" cy="130" rx="15" ry="10" fill="white" />
                            </g>
                        )}
                    </svg>
                );

            case 'cloudy':
                return (
                    <svg viewBox="0 0 200 200" style={{ width: '100%', height: '100%' }}>
                        <defs>
                            <linearGradient id="cloudGrad1" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" stopColor="#E8DFF5" />
                                <stop offset="100%" stopColor="#D8C7FF" />
                            </linearGradient>
                            <linearGradient id="cloudGrad2" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" stopColor="white" />
                                <stop offset="100%" stopColor="#F0E8FF" />
                            </linearGradient>
                        </defs>
                        {/* Background cloud */}
                        <g style={{ animation: 'drift-right 10s ease-in-out infinite' }}>
                            <ellipse cx="120" cy="95" rx="50" ry="28" fill="url(#cloudGrad1)" opacity="0.7" />
                            <ellipse cx="100" cy="85" rx="35" ry="24" fill="url(#cloudGrad1)" opacity="0.7" />
                            <ellipse cx="140" cy="88" rx="30" ry="20" fill="url(#cloudGrad1)" opacity="0.7" />
                        </g>
                        {/* Main cloud */}
                        <g style={{ animation: 'drift-right 7s ease-in-out infinite' }}>
                            <ellipse cx="105" cy="115" rx="55" ry="30" fill="url(#cloudGrad2)" />
                            <ellipse cx="80" cy="105" rx="38" ry="26" fill="url(#cloudGrad2)" />
                            <ellipse cx="130" cy="108" rx="32" ry="22" fill="url(#cloudGrad2)" />
                            <ellipse cx="100" cy="100" rx="28" ry="22" fill="white" />
                        </g>
                        {/* Peek of sun */}
                        <circle cx="160" cy="70" r="22" fill="#FFD6B0" opacity="0.4" />
                    </svg>
                );

            case 'breezy':
                return (
                    <svg viewBox="0 0 200 200" style={{ width: '100%', height: '100%' }}>
                        {/* Breeze lines */}
                        {[0, 1, 2, 3, 4].map((i) => (
                            <path
                                key={i}
                                d={`M${30 + i * 8},${80 + i * 20} Q${100 + i * 5},${70 + i * 20} ${170 - i * 5},${85 + i * 20}`}
                                fill="none"
                                stroke="#D8C7FF"
                                strokeWidth="2"
                                strokeLinecap="round"
                                opacity={0.3 + i * 0.1}
                                style={{
                                    animation: `drift-right ${5 + i}s ease-in-out infinite`,
                                    animationDelay: `${i * 0.5}s`,
                                }}
                            />
                        ))}
                        {/* Small cloud */}
                        <g style={{ animation: 'drift-right 4s ease-in-out infinite' }}>
                            <ellipse cx="100" cy="90" rx="40" ry="20" fill="white" opacity="0.7" />
                            <ellipse cx="85" cy="84" rx="25" ry="16" fill="white" opacity="0.7" />
                            <ellipse cx="115" cy="86" rx="22" ry="14" fill="white" opacity="0.7" />
                        </g>
                    </svg>
                );

            case 'recovery':
                return (
                    <svg viewBox="0 0 200 200" style={{ width: '100%', height: '100%' }}>
                        <defs>
                            <linearGradient id="rainbowGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                                <stop offset="0%" stopColor="#F8C8DC" />
                                <stop offset="25%" stopColor="#FFD6B0" />
                                <stop offset="50%" stopColor="#BEE3F8" />
                                <stop offset="75%" stopColor="#D8C7FF" />
                                <stop offset="100%" stopColor="#F8C8DC" />
                            </linearGradient>
                        </defs>
                        {/* Rainbow arc */}
                        <path
                            d="M30,140 Q100,40 170,140"
                            fill="none"
                            stroke="url(#rainbowGrad)"
                            strokeWidth="8"
                            strokeLinecap="round"
                            opacity="0.6"
                            style={{ animation: 'breathe 5s ease-in-out infinite' }}
                        />
                        <path
                            d="M40,140 Q100,50 160,140"
                            fill="none"
                            stroke="url(#rainbowGrad)"
                            strokeWidth="5"
                            strokeLinecap="round"
                            opacity="0.4"
                        />
                        {/* Small sun peeking */}
                        <circle cx="100" cy="75" r="20" fill="#FFD6B0" opacity="0.5">
                            <animate attributeName="opacity" values="0.3;0.6;0.3" dur="4s" repeatCount="indefinite" />
                        </circle>
                    </svg>
                );

            default:
                return null;
        }
    };

    return (
        <div style={containerStyle}>
            <div style={glowStyle} />
            {renderIcon()}
        </div>
    );
}
