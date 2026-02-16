import { useState } from 'react';
import { type CSSProperties } from 'react';

const pageStyle: CSSProperties = {
  padding: 'var(--space-3xl) var(--space-lg) var(--space-xl)',
  animation: 'fade-in-up 0.8s ease-out',
};

const prompts = [
  'What felt gentle about today?',
  'What is your body asking for right now?',
  'When did you last feel truly at ease?',
  'What tiny thing brought you comfort today?',
  'If your feelings were weather, what would the sky look like right now?',
];

const pastEntries = [
  {
    date: 'Yesterday',
    weather: 'Soft Sunshine',
    note: 'I took a longer walk than usual and noticed how the light felt warm on my skin. Small things felt enough.',
  },
  {
    date: '2 days ago',
    weather: 'Light Winds',
    note: 'A busier day — I felt the wind picking up but remembered to pause. Tea helped.',
  },
  {
    date: '4 days ago',
    weather: 'Gentle Clearing',
    note: 'Something shifted after the rain. I felt lighter by evening, like the air after a storm.',
  },
];

export default function ReflectionPage() {
  const [currentPrompt] = useState(() => prompts[Math.floor(Math.random() * prompts.length)]);
  const [entry, setEntry] = useState('');

  const textareaStyle: CSSProperties = {
    width: '100%',
    minHeight: '120px',
    border: 'none',
    borderRadius: 'var(--radius-md)',
    padding: 'var(--space-lg)',
    fontFamily: 'var(--font-body)',
    fontSize: '0.92rem',
    fontWeight: 300,
    color: 'var(--text-primary)',
    background: 'rgba(255, 255, 255, 0.35)',
    backdropFilter: 'blur(8px)',
    outline: 'none',
    resize: 'vertical',
    lineHeight: 1.7,
    letterSpacing: '0.01em',
  };

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
        A Quiet Moment
      </p>
      <h1 style={{
        fontFamily: 'var(--font-heading)',
        fontSize: '1.8rem',
        fontWeight: 500,
        fontStyle: 'italic',
        color: 'var(--text-primary)',
        marginBottom: 'var(--space-sm)',
      }}>
        Reflection
      </h1>
      <p style={{
        fontFamily: 'var(--font-body)',
        fontSize: '0.85rem',
        color: 'var(--text-secondary)',
        fontWeight: 300,
        marginBottom: 'var(--space-2xl)',
        lineHeight: 1.6,
      }}>
        No right answers here — only soft observations.
      </p>

      {/* Today's prompt */}
      <div className="glass-card" style={{
        padding: 'var(--space-xl)',
        marginBottom: 'var(--space-xl)',
      }}>
        <p style={{
          fontFamily: 'var(--font-heading)',
          fontSize: '1.15rem',
          fontWeight: 500,
          fontStyle: 'italic',
          color: 'var(--text-primary)',
          marginBottom: 'var(--space-lg)',
          lineHeight: 1.5,
        }}>
          "{currentPrompt}"
        </p>

        <textarea
          style={textareaStyle}
          placeholder="Write softly here..."
          value={entry}
          onChange={(e) => setEntry(e.target.value)}
        />

        <div style={{
          display: 'flex',
          justifyContent: 'flex-end',
          marginTop: 'var(--space-md)',
        }}>
          <button style={{
            border: 'none',
            borderRadius: 'var(--radius-full)',
            padding: 'var(--space-sm) var(--space-xl)',
            background: 'linear-gradient(135deg, var(--lavender), var(--blush))',
            color: 'var(--text-primary)',
            fontFamily: 'var(--font-body)',
            fontSize: '0.8rem',
            fontWeight: 500,
            letterSpacing: '0.05em',
            cursor: 'pointer',
            transition: 'all 0.3s ease',
            opacity: entry.trim() ? 1 : 0.5,
          }}>
            Save This Moment
          </button>
        </div>
      </div>

      {/* Past reflections */}
      <p style={{
        fontFamily: 'var(--font-body)',
        fontSize: '0.7rem',
        fontWeight: 500,
        letterSpacing: '0.12em',
        textTransform: 'uppercase',
        color: 'var(--text-light)',
        marginBottom: 'var(--space-md)',
      }}>
        Recent Reflections
      </p>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
        {pastEntries.map((e, i) => (
          <div key={i} className="glass-card" style={{
            padding: 'var(--space-lg)',
            animation: `fade-in-up 0.8s ease-out ${0.2 + i * 0.15}s both`,
          }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: 'var(--space-sm)',
            }}>
              <span style={{
                fontSize: '0.7rem',
                color: 'var(--text-light)',
                fontWeight: 500,
                letterSpacing: '0.05em',
              }}>
                {e.date}
              </span>
              <span style={{
                fontSize: '0.72rem',
                color: 'var(--text-secondary)',
                fontWeight: 500,
                fontStyle: 'italic',
                fontFamily: 'var(--font-heading)',
              }}>
                {e.weather}
              </span>
            </div>
            <p style={{
              fontFamily: 'var(--font-body)',
              fontSize: '0.88rem',
              fontWeight: 300,
              color: 'var(--text-secondary)',
              lineHeight: 1.7,
            }}>
              {e.note}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
