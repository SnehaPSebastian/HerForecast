import { useState } from 'react';
import { type CSSProperties } from 'react';

const pageStyle: CSSProperties = {
  padding: 'var(--space-3xl) var(--space-lg) var(--space-xl)',
  animation: 'fade-in-up 0.8s ease-out',
};

interface ToggleProps {
  label: string;
  description: string;
  defaultOn?: boolean;
}

function SettingToggle({ label, description, defaultOn = false }: ToggleProps) {
  const [on, setOn] = useState(defaultOn);

  return (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: 'var(--space-lg) 0',
      borderBottom: '1px solid rgba(216, 199, 255, 0.15)',
    }}>
      <div style={{ flex: 1, paddingRight: 'var(--space-lg)' }}>
        <p style={{
          fontFamily: 'var(--font-body)',
          fontSize: '0.92rem',
          fontWeight: 500,
          color: 'var(--text-primary)',
          marginBottom: '4px',
        }}>
          {label}
        </p>
        <p style={{
          fontFamily: 'var(--font-body)',
          fontSize: '0.78rem',
          fontWeight: 300,
          color: 'var(--text-secondary)',
          lineHeight: 1.5,
        }}>
          {description}
        </p>
      </div>
      <button
        onClick={() => setOn(!on)}
        style={{
          width: '48px',
          height: '28px',
          borderRadius: 'var(--radius-full)',
          border: 'none',
          background: on
            ? 'linear-gradient(135deg, var(--lavender), var(--blush))'
            : 'rgba(200, 190, 220, 0.25)',
          cursor: 'pointer',
          position: 'relative',
          transition: 'background 0.3s ease',
          flexShrink: 0,
        }}
      >
        <div style={{
          width: '22px',
          height: '22px',
          borderRadius: '50%',
          background: 'white',
          boxShadow: '0 2px 6px rgba(0,0,0,0.1)',
          position: 'absolute',
          top: '3px',
          left: on ? '23px' : '3px',
          transition: 'left 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)',
        }} />
      </button>
    </div>
  );
}

export default function SettingsPage() {

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
        Your Space
      </p>
      <h1 style={{
        fontFamily: 'var(--font-heading)',
        fontSize: '1.8rem',
        fontWeight: 500,
        fontStyle: 'italic',
        color: 'var(--text-primary)',
        marginBottom: 'var(--space-sm)',
      }}>
        Settings
      </h1>
      <p style={{
        fontFamily: 'var(--font-body)',
        fontSize: '0.85rem',
        color: 'var(--text-secondary)',
        fontWeight: 300,
        marginBottom: 'var(--space-2xl)',
        lineHeight: 1.6,
      }}>
        Customize how your inner weather feels.
      </p>

      {/* Toggles */}
      <div className="glass-card" style={{
        padding: 'var(--space-lg) var(--space-xl)',
        marginBottom: 'var(--space-xl)',
      }}>
        <SettingToggle
          label="Gentle Reminders"
          description="Receive soft nudges to check in with yourself throughout the day."
          defaultOn={true}
        />
        <SettingToggle
          label="Floating Particles"
          description="Show ambient sparkles and pollen in the background."
          defaultOn={true}
        />
        <SettingToggle
          label="Sound Landscapes"
          description="Play soft ambient audio that matches your current weather state."
          defaultOn={false}
        />
        <SettingToggle
          label="Cycle Tracking"
          description="Include cycle-aware insights in your daily weather."
          defaultOn={false}
        />
      </div>

      {/* About */}
      <div className="glass-card" style={{
        padding: 'var(--space-xl)',
        textAlign: 'center',
      }}>
        <p style={{
          fontFamily: 'var(--font-heading)',
          fontSize: '1.1rem',
          fontWeight: 500,
          fontStyle: 'italic',
          color: 'var(--text-primary)',
          marginBottom: 'var(--space-sm)',
        }}>
          Inner Weather
        </p>
        <p style={{
          fontFamily: 'var(--font-body)',
          fontSize: '0.78rem',
          color: 'var(--text-light)',
          fontWeight: 300,
        }}>
          Your body speaks in weather.{'\n'}We just help you listen.
        </p>
      </div>
    </div>
  );
}
