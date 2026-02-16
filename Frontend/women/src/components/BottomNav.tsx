import { type CSSProperties } from 'react';

interface NavItem {
    id: string;
    label: string;
    icon: React.ReactNode;
}

const navItems: NavItem[] = [
    {
        id: 'home',
        label: 'Home',
        icon: (
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="5" />
                <line x1="12" y1="1" x2="12" y2="4" />
                <line x1="12" y1="20" x2="12" y2="23" />
                <line x1="4.22" y1="4.22" x2="6.34" y2="6.34" />
                <line x1="17.66" y1="17.66" x2="19.78" y2="19.78" />
                <line x1="1" y1="12" x2="4" y2="12" />
                <line x1="20" y1="12" x2="23" y2="12" />
                <line x1="4.22" y1="19.78" x2="6.34" y2="17.66" />
                <line x1="17.66" y1="6.34" x2="19.78" y2="4.22" />
            </svg>
        ),
    },
    {
        id: 'trends',
        label: 'Trends',
        icon: (
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M3 20Q8 10 12 14Q16 18 21 6" />
            </svg>
        ),
    },
    {
        id: 'reflection',
        label: 'Reflection',
        icon: (
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 3C7 3 3 7 3 12C3 14 4 16 5 17L3 21L7 19C9 20 10 21 12 21C17 21 21 17 21 12C21 7 17 3 12 3Z" />
                <circle cx="8" cy="12" r="0.5" fill="currentColor" />
                <circle cx="12" cy="12" r="0.5" fill="currentColor" />
                <circle cx="16" cy="12" r="0.5" fill="currentColor" />
            </svg>
        ),
    },
    {
        id: 'settings',
        label: 'Settings',
        icon: (
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="3" />
                <path d="M12 1V3M12 21V23M4.22 4.22L5.64 5.64M18.36 18.36L19.78 19.78M1 12H3M21 12H23M4.22 19.78L5.64 18.36M18.36 5.64L19.78 4.22" />
            </svg>
        ),
    },
    {
        id: 'simulation',
        label: 'Simulate',
        icon: (
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                <line x1="9" y1="9" x2="15" y2="15" />
                <line x1="15" y1="9" x2="9" y2="15" />
            </svg>
        ),
    },
];

interface BottomNavProps {
    activePage: string;
    onNavigate: (page: string) => void;
}

export default function BottomNav({ activePage, onNavigate }: BottomNavProps) {
    const navStyle: CSSProperties = {
        position: 'fixed',
        bottom: 0,
        left: '50%',
        transform: 'translateX(-50%)',
        width: '100%',
        maxWidth: '960px',
        padding: 'var(--space-sm) var(--space-lg) calc(var(--space-md) + env(safe-area-inset-bottom))',
        background: 'rgba(255, 255, 255, 0.7)',
        backdropFilter: 'blur(24px)',
        WebkitBackdropFilter: 'blur(24px)',
        borderTop: '1px solid rgba(255, 255, 255, 0.8)',
        display: 'flex',
        justifyContent: 'space-around',
        alignItems: 'center',
        zIndex: 100,
    };

    const buttonBase: CSSProperties = {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '4px',
        border: 'none',
        background: 'none',
        cursor: 'pointer',
        padding: 'var(--space-sm)',
        borderRadius: 'var(--radius-md)',
        transition: 'all 0.3s ease',
        minWidth: '56px',
    };

    return (
        <nav style={navStyle}>
            {navItems.map((item) => {
                const isActive = activePage === item.id;
                return (
                    <button
                        key={item.id}
                        onClick={() => onNavigate(item.id)}
                        style={{
                            ...buttonBase,
                            color: isActive ? 'var(--text-primary)' : 'var(--text-light)',
                            stroke: isActive ? 'var(--text-primary)' : 'var(--text-light)',
                            background: isActive ? 'rgba(216, 199, 255, 0.2)' : 'transparent',
                            transform: isActive ? 'scale(1.05)' : 'scale(1)',
                        }}
                    >
                        {item.icon}
                        <span style={{
                            fontSize: '0.6rem',
                            fontWeight: isActive ? 600 : 400,
                            letterSpacing: '0.05em',
                            fontFamily: 'var(--font-body)',
                        }}>
                            {item.label}
                        </span>
                    </button>
                );
            })}
        </nav>
    );
}
