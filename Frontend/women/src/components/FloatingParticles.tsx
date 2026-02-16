import { useMemo } from 'react';
import { type CSSProperties } from 'react';

interface Particle {
    id: number;
    left: string;
    size: number;
    duration: number;
    delay: number;
    opacity: number;
}

export default function FloatingParticles() {
    const particles = useMemo<Particle[]>(() => {
        return Array.from({ length: 20 }, (_, i) => ({
            id: i,
            left: `${Math.random() * 100}%`,
            size: 3 + Math.random() * 6,
            duration: 12 + Math.random() * 18,
            delay: Math.random() * 15,
            opacity: 0.15 + Math.random() * 0.35,
        }));
    }, []);

    const containerStyle: CSSProperties = {
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        overflow: 'hidden',
        zIndex: 0,
    };

    return (
        <div style={containerStyle} aria-hidden="true">
            {particles.map((p) => (
                <div
                    key={p.id}
                    style={{
                        position: 'absolute',
                        left: p.left,
                        bottom: '-20px',
                        width: `${p.size}px`,
                        height: `${p.size}px`,
                        borderRadius: '50%',
                        background: `radial-gradient(circle, rgba(216,199,255,${p.opacity}) 0%, rgba(248,200,220,${p.opacity * 0.6}) 60%, transparent 100%)`,
                        animation: `drift-up ${p.duration}s linear ${p.delay}s infinite`,
                        willChange: 'transform, opacity',
                    }}
                />
            ))}

            {/* Sparkles */}
            {Array.from({ length: 8 }, (_, i) => (
                <div
                    key={`sparkle-${i}`}
                    style={{
                        position: 'absolute',
                        left: `${10 + Math.random() * 80}%`,
                        top: `${10 + Math.random() * 70}%`,
                        width: '4px',
                        height: '4px',
                        borderRadius: '50%',
                        background: 'rgba(255, 255, 255, 0.8)',
                        boxShadow: '0 0 6px rgba(216, 199, 255, 0.5)',
                        animation: `sparkle ${3 + Math.random() * 4}s ease-in-out ${Math.random() * 5}s infinite`,
                    }}
                />
            ))}
        </div>
    );
}
