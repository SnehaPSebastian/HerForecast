import { useState } from 'react';
import { type CyclePhase } from '../data/cycleData';
import { simulatePrediction, type PredictRequest, type SimulationResponse } from '../api/phaseApi';

interface SimulationPageProps {
  onCyclePhaseChange: (phase: CyclePhase) => void;
}

export default function SimulationPage({ onCyclePhaseChange }: SimulationPageProps) {
  const [file, setFile] = useState<File | null>(null);
  const [results, setResults] = useState<SimulationResponse[]>([]);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
      setResults([]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    setResults([]);

    const reader = new FileReader();
    reader.onload = async (event) => {
      const csvText = event.target?.result as string;
      if (!csvText) return;

      const lines = csvText.trim().split('\n');
      if (lines.length < 2) {
        setResults([]);
        setLoading(false);
        return;
      }

      const headers = lines[0].split(',').map(h => h.trim());
      const requiredHeaders = ['spo2', 'gsr_mean', 'gsr_phasic_std', 'ppg_rmssd', 'heart_rate', 'skin_temp', 'estrogen', 'progesterone', 'day_in_cycle'];

      const headerIndices: { [key: string]: number } = {};
      for (const req of requiredHeaders) {
        const index = headers.indexOf(req);
        if (index === -1) {
          alert(`Missing required column: ${req}`);
          setLoading(false);
          return;
        }
        headerIndices[req] = index;
      }

      const predictions: SimulationResponse[] = [];

      for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',').map(v => v.trim());
        if (values.length !== headers.length) continue;

        const request: PredictRequest = {
          wearable_data: {
            spo2: parseFloat(values[headerIndices.spo2]),
            gsr_mean: parseFloat(values[headerIndices.gsr_mean]),
            gsr_phasic_std: parseFloat(values[headerIndices.gsr_phasic_std]),
            ppg_rmssd: parseFloat(values[headerIndices.ppg_rmssd]),
            heart_rate: parseFloat(values[headerIndices.heart_rate]),
            skin_temp: parseFloat(values[headerIndices.skin_temp]),
          },
          hormone_data: {
            estrogen: parseFloat(values[headerIndices.estrogen]),
            progesterone: parseFloat(values[headerIndices.progesterone]),
          },
          day_in_cycle: parseInt(values[headerIndices.day_in_cycle]),
        };

        try {
          const res = await simulatePrediction(request);
          predictions.push(res);
        } catch (error) {
          console.error(`Error processing row ${i}:`, error);
        }
      }

      setResults(predictions);
      setLoading(false);

      // Automatically apply the theme of the first prediction
      if (predictions.length > 0) {
        onCyclePhaseChange(predictions[0].predicted_phase.toLowerCase() as CyclePhase);
      }
    };

    reader.readAsText(file);
  };

  return (
    <div style={{ padding: 'var(--space-lg)', maxWidth: '600px', margin: '0 auto' }}>
      <h1 style={{
        fontFamily: 'var(--font-heading)',
        fontSize: '1.5rem',
        color: 'var(--text-primary)',
        marginBottom: 'var(--space-lg)',
        textAlign: 'center',
      }}>
        Model Testing - File Upload
      </h1>

      <div style={{
        background: 'rgba(255, 255, 255, 0.1)',
        backdropFilter: 'blur(10px)',
        borderRadius: 'var(--radius-lg)',
        padding: 'var(--space-lg)',
        marginBottom: 'var(--space-lg)',
      }}>
        <h2 style={{ fontFamily: 'var(--font-body)', fontSize: '1rem', color: 'var(--text-primary)', marginBottom: 'var(--space-sm)' }}>
          Required CSV File Structure
        </h2>
        <p style={{ fontFamily: 'var(--font-body)', fontSize: '0.9rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
          Upload a CSV file for testing the model. The file must contain the following columns matching the model's input features:
        </p>
        <ul style={{ fontFamily: 'var(--font-body)', fontSize: '0.9rem', color: 'var(--text-secondary)', marginTop: 'var(--space-sm)' }}>
          <li><strong>spo2</strong>: Blood oxygen saturation (90-100)</li>
          <li><strong>gsr_mean</strong>: Galvanic skin response mean (0-10)</li>
          <li><strong>gsr_phasic_std</strong>: GSR phasic standard deviation (0-1)</li>
          <li><strong>ppg_rmssd</strong>: Photoplethysmography RMSSD (10-100)</li>
          <li><strong>heart_rate</strong>: Heart rate in BPM (50-150)</li>
          <li><strong>skin_temp</strong>: Skin temperature in Celsius (30-40)</li>
          <li><strong>estrogen</strong>: Estrogen level (0-500)</li>
          <li><strong>progesterone</strong>: Progesterone level (0-50)</li>
          <li><strong>day_in_cycle</strong>: Day in menstrual cycle (1-28)</li>
        </ul>
        <p style={{ fontFamily: 'var(--font-body)', fontSize: '0.9rem', color: 'var(--text-secondary)', marginTop: 'var(--space-sm)' }}>
          Each row represents a data point for prediction. The model will process the file and return phase predictions for each entry.
        </p>
      </div>

      <form onSubmit={handleSubmit} style={{
        background: 'rgba(255, 255, 255, 0.1)',
        backdropFilter: 'blur(10px)',
        borderRadius: 'var(--radius-lg)',
        padding: 'var(--space-lg)',
        display: 'flex',
        flexDirection: 'column',
        gap: 'var(--space-md)',
      }}>
        <label>
          Select CSV File:
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            required
            style={{
              marginTop: 'var(--space-sm)',
              padding: 'var(--space-sm)',
              borderRadius: 'var(--radius-md)',
              border: '1px solid var(--text-secondary)',
              background: 'rgba(255, 255, 255, 0.1)',
              color: 'var(--text-primary)',
            }}
          />
        </label>

        <button
          type="submit"
          disabled={!file || loading}
          style={{
            background: 'var(--lavender)',
            color: 'var(--text-primary)',
            border: 'none',
            borderRadius: 'var(--radius-md)',
            padding: 'var(--space-sm) var(--space-md)',
            fontFamily: 'var(--font-body)',
            cursor: (!file || loading) ? 'not-allowed' : 'pointer',
          }}
        >
          {loading ? 'Processing...' : 'Upload and Test Model'}
        </button>
      </form>

      {results.length > 0 && (
        <div style={{
          marginTop: 'var(--space-lg)',
          background: 'rgba(255, 255, 255, 0.1)',
          backdropFilter: 'blur(10px)',
          borderRadius: 'var(--radius-lg)',
          padding: 'var(--space-lg)',
        }}>
          <h2 style={{ fontFamily: 'var(--font-heading)', color: 'var(--text-primary)' }}>
            Prediction Results ({results.length} entries)
          </h2>
          <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
            {results.map((res, index) => (
              <div key={index} style={{
                marginBottom: 'var(--space-md)',
                padding: 'var(--space-sm)',
                background: 'rgba(255, 255, 255, 0.05)',
                borderRadius: 'var(--radius-sm)',
              }}>
                <p><strong>Row {index + 1}:</strong> Phase: {res.predicted_phase}, Mood: {res.predicted_mood}, Confidence: {Math.round(res.confidence * 100)}%</p>
                <button
                  onClick={() => onCyclePhaseChange(res.predicted_phase.toLowerCase() as CyclePhase)}
                  style={{
                    marginTop: 'var(--space-sm)',
                    background: 'var(--lavender)',
                    color: 'var(--text-primary)',
                    border: 'none',
                    borderRadius: 'var(--radius-sm)',
                    padding: '4px 8px',
                    fontSize: '0.8rem',
                    cursor: 'pointer',
                  }}
                >
                  Apply Theme
                </button>
                <details>
                  <summary>Probabilities</summary>
                  <ul>
                    {Object.entries(res.probabilities).map(([phase, prob]) => (
                      <li key={phase}>{phase}: {Math.round(prob * 100)}%</li>
                    ))}
                  </ul>
                </details>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
