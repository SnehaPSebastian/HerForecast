import WeatherHero from '../components/WeatherHero';
import InsightsSection from '../components/InsightsSection';
import ForecastSection from '../components/ForecastSection';
import CycleBanner from '../components/CycleBanner';
import { cyclePhases, type CyclePhase } from '../data/cycleData';

interface PredictionInfo {
    confidence: number;
    dayInStudy: number;
    userId: number;
    source: 'api' | 'fallback';
}

interface HomePageProps {
    cyclePhase: CyclePhase;
    onCyclePhaseChange: (phase: CyclePhase) => void;
    predictionInfo: PredictionInfo | null;
    availableUsers: number[];
    selectedUser: number | null;
    onUserChange: (id: number) => void;
}

export default function HomePage({
    cyclePhase,
    onCyclePhaseChange,
    predictionInfo,
    availableUsers,
    selectedUser,
    onUserChange,
}: HomePageProps) {
    const data = cyclePhases[cyclePhase];

    return (
        <>
            {/* ML Prediction Badge */}
            {predictionInfo && predictionInfo.source === 'api' && (
                <div style={{
                    display: 'flex',
                    justifyContent: 'center',
                    paddingTop: 'var(--space-md)',
                    animation: 'fade-in-up 0.6s ease-out both',
                }}>
                    <div style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '8px',
                        background: 'rgba(255, 255, 255, 0.4)',
                        backdropFilter: 'blur(12px)',
                        borderRadius: 'var(--radius-full)',
                        padding: '6px 16px',
                        fontSize: '0.65rem',
                        fontFamily: 'var(--font-body)',
                        color: 'var(--text-light)',
                        letterSpacing: '0.05em',
                    }}>
                        <span style={{
                            width: '6px',
                            height: '6px',
                            borderRadius: '50%',
                            background: '#7CD4A0',
                            animation: 'breathe 3s ease-in-out infinite',
                        }} />
                        <span>Predicted by AI</span>
                        <span style={{ opacity: 0.5 }}>·</span>
                        <span>{Math.round(predictionInfo.confidence * 100)}% confidence</span>
                    </div>
                </div>
            )}

            {/* User Selector */}
            {availableUsers.length > 1 && (
                <div style={{
                    display: 'flex',
                    justifyContent: 'center',
                    padding: 'var(--space-sm) var(--space-lg) 0',
                }}>
                    <select
                        value={selectedUser ?? ''}
                        onChange={(e) => onUserChange(Number(e.target.value))}
                        style={{
                            background: 'rgba(255, 255, 255, 0.35)',
                            backdropFilter: 'blur(8px)',
                            border: '1px solid rgba(255, 255, 255, 0.5)',
                            borderRadius: 'var(--radius-md)',
                            padding: '6px 12px',
                            fontFamily: 'var(--font-body)',
                            fontSize: '0.7rem',
                            color: 'var(--text-secondary)',
                            cursor: 'pointer',
                            outline: 'none',
                        }}
                    >
                        {availableUsers.map((id) => (
                            <option key={id} value={id}>
                                User {id}
                            </option>
                        ))}
                    </select>
                </div>
            )}

            <WeatherHero
                state={data.weatherState}
                title={data.heroTitle}
                message={data.heroMessage}
            />

            <CycleBanner
                phaseData={data}
                currentPhase={cyclePhase}
                onPhaseChange={onCyclePhaseChange}
            />

            <InsightsSection
                nervousSystem={data.nervousSystem}
                emotionalLoad={data.emotionalLoad}
                recovery={data.recovery}
            />

            <ForecastSection
                title={data.forecastTitle}
                message={data.forecastMessage}
            />
        </>
    );
}
