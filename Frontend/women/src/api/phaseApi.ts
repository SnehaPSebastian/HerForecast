import { type CyclePhase } from '../data/cycleData';

const API_BASE = 'http://localhost:8001';

export interface PredictionResponse {
    phase: CyclePhase;
    predicted_raw: string;
    true_raw: string;
    confidence: number;
    day_in_study: number;
    user_id: number;
    message?: string;
}

export interface UserTimelineEntry {
    day: number;
    predicted: CyclePhase;
    actual: CyclePhase;
}

export interface TimelineResponse {
    user_id: number;
    total_days: number;
    timeline: UserTimelineEntry[];
}

export interface WearableData {
  spo2: number;
  gsr_mean: number;
  gsr_phasic_std: number;
  ppg_rmssd: number;
  heart_rate: number;
  skin_temp: number;
}

export interface HormoneData {
  estrogen: number;
  progesterone: number;
}

export interface PredictRequest {
  wearable_data: WearableData;
  hormone_data: HormoneData;
  day_in_cycle: number;
}

export interface SimulationResponse {
  predicted_phase: string;
  predicted_mood: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface ModelInfoResponse {
    model_name: string;
    expected_features: string[];
}

/**
 * Fetch the predicted cycle phase for a given user (and optional day).
 */
export async function fetchPrediction(userId: number, day?: number): Promise<PredictionResponse> {
    const params = new URLSearchParams({ id: String(userId) });
    if (day !== undefined) params.set('day', String(day));

    const res = await fetch(`${API_BASE}/predict?${params}`);
    if (!res.ok) throw new Error(`Prediction request failed: ${res.status}`);
    return res.json();
}

/**
 * Fetch the list of available user IDs.
 */
export async function fetchUsers(): Promise<number[]> {
    const res = await fetch(`${API_BASE}/users`);
    if (!res.ok) throw new Error(`Users request failed: ${res.status}`);
    const data = await res.json();
    return data.users;
}

/**
 * Fetch the full phase timeline for a user.
 */
export async function fetchTimeline(userId: number): Promise<TimelineResponse> {
    const res = await fetch(`${API_BASE}/user/${userId}/timeline`);
    if (!res.ok) throw new Error(`Timeline request failed: ${res.status}`);
    return res.json();
}

/**
 * Fetch model performance info.
 */
export async function fetchModelInfo(): Promise<ModelInfoResponse> {
    const res = await fetch(`${API_BASE}/model/info`);
    if (!res.ok) throw new Error(`Model info request failed: ${res.status}`);
    return res.json();
}

/**
 * Simulate prediction with custom wearable and hormone data.
 */
export async function simulatePrediction(data: PredictRequest): Promise<SimulationResponse> {
    const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error(`Simulation request failed: ${res.status}`);
    return res.json();
}
