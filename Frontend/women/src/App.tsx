import { useState, useEffect } from 'react';
import './index.css';
import FloatingParticles from './components/FloatingParticles';
import BottomNav from './components/BottomNav';
import HomePage from './pages/HomePage';
import TrendsPage from './pages/TrendsPage';
import ReflectionPage from './pages/ReflectionPage';
import SettingsPage from './pages/SettingsPage';
import SimulationPage from './pages/SimulationPage';
import { cyclePhases, type CyclePhase } from './data/cycleData';
import { fetchPrediction, fetchUsers } from './api/phaseApi';

type WeatherState = 'clearing' | 'calm' | 'cloudy' | 'breezy' | 'recovery';

const weatherGradients: Record<WeatherState, string> = {
  clearing: 'linear-gradient(170deg, #EDE4FF 0%, #F8C8DC 25%, #FFF9F5 50%, #FFF5EE 100%)',
  calm: 'linear-gradient(170deg, #FFF5E6 0%, #FFE8CC 20%, #FFD6B0 45%, #FFF9F5 100%)',
  cloudy: 'linear-gradient(170deg, #E0D6F0 0%, #D8C7FF 25%, #E8E0F5 55%, #F0EAF8 100%)',
  breezy: 'linear-gradient(170deg, #D6EAF8 0%, #BEE3F8 25%, #E8F4FD 50%, #F5FAFF 100%)',
  recovery: 'linear-gradient(170deg, #F8E8F0 0%, #F8C8DC 20%, #FFD6B0 45%, #E8F4FD 70%, #EDE4FF 100%)',
};

export default function App() {
  const [page, setPage] = useState('home');
  const [cyclePhase, setCyclePhase] = useState<CyclePhase>('follicular');
  const [isLoading, setIsLoading] = useState(true);
  const [predictionInfo, setPredictionInfo] = useState<{
    confidence: number;
    dayInStudy: number;
    userId: number;
    source: 'api' | 'fallback';
  } | null>(null);
  const [availableUsers, setAvailableUsers] = useState<number[]>([]);
  const [selectedUser, setSelectedUser] = useState<number | null>(null);

  // Fetch available users and initial prediction on mount
  useEffect(() => {
    async function loadData() {
      try {
        const users = await fetchUsers();
        setAvailableUsers(users);

        // Pick the first user by default
        const userId = users[0] ?? 6;
        setSelectedUser(userId);

        const prediction = await fetchPrediction(userId);
        setCyclePhase(prediction.phase.toLowerCase() as CyclePhase);
        setPredictionInfo({
          confidence: prediction.confidence,
          dayInStudy: prediction.day_in_study,
          userId: prediction.user_id,
          source: 'api',
        });
      } catch {
        // Backend not running — fall back to default
        setPredictionInfo({
          confidence: 0,
          dayInStudy: 0,
          userId: 0,
          source: 'fallback',
        });
      } finally {
        setIsLoading(false);
      }
    }
    loadData();
  }, []);

  // Re-fetch prediction when user changes
  useEffect(() => {
    if (selectedUser === null) return;
    async function reload() {
      try {
        const prediction = await fetchPrediction(selectedUser!);
        setCyclePhase(prediction.phase.toLowerCase() as CyclePhase);
        setPredictionInfo({
          confidence: prediction.confidence,
          dayInStudy: prediction.day_in_study,
          userId: prediction.user_id,
          source: 'api',
        });
      } catch {
        // Keep current phase on failure
      }
    }
    reload();
  }, [selectedUser]);

  const currentWeather = cyclePhases[cyclePhase].weatherState;

  const renderPage = () => {
    switch (page) {
      case 'home':
        return (
          <HomePage
            cyclePhase={cyclePhase}
            onCyclePhaseChange={setCyclePhase}
            predictionInfo={predictionInfo}
            availableUsers={availableUsers}
            selectedUser={selectedUser}
            onUserChange={setSelectedUser}
          />
        );
      case 'trends':
        return <TrendsPage />;
      case 'reflection':
        return <ReflectionPage />;
      case 'settings':
        return <SettingsPage />;
      case 'simulation':
        return <SimulationPage onCyclePhaseChange={setCyclePhase} />;
      default:
        return (
          <HomePage
            cyclePhase={cyclePhase}
            onCyclePhaseChange={setCyclePhase}
            predictionInfo={predictionInfo}
            availableUsers={availableUsers}
            selectedUser={selectedUser}
            onUserChange={setSelectedUser}
          />
        );
    }
  };

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      background: weatherGradients[currentWeather],
      transition: 'background 1.2s ease-in-out',
      overflowY: 'auto',
      overflowX: 'hidden',
    }}>
      <div id="inner-root" style={{
        position: 'relative',
        minHeight: '100vh',
        maxWidth: '600px',
        margin: '0 auto',
        padding: '0 var(--space-md) 90px',
      }}>
        <FloatingParticles />
        <main key={page} style={{ position: 'relative', zIndex: 1 }}>
          {isLoading ? (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              height: '60vh',
              fontFamily: 'var(--font-heading)',
              fontSize: '1.2rem',
              color: 'var(--text-light)',
              fontStyle: 'italic',
              animation: 'breathe 3s ease-in-out infinite',
            }}>
              Sensing your weather…
            </div>
          ) : (
            renderPage()
          )}
        </main>
        <BottomNav activePage={page} onNavigate={setPage} />
      </div>
    </div>
  );
}
