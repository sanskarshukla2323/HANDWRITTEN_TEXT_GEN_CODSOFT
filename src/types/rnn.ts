/**
 * Types for Handwriting Studio - Character-Level RNN (CODSOFT Task 2)
 */

export interface CharVocab {
  chars: string[];
  charToIdx: Record<string, number>;
  idxToChar: Record<number, string>;
  vocabSize: number;
}

export interface DatasetStats {
  totalChars: number;
  uniqueChars: number;
  lineCount: number;
  wordCount: number;
  alphabeticCount: number;
  digitCount: number;
  whitespaceCount: number;
  punctuationCount: number;
}

export interface TrainingConfig {
  architecture: 'LSTM' | 'SimpleRNN';
  hiddenUnits: number;
  sequenceLength: number;
  learningRate: number;
  batchSize: number;
  epochs: number;
}

export interface TrainingMetric {
  epoch: number;
  loss: number;
  perplexity: number;
  elapsedMs: number;
  sampleGeneration?: string;
}

export interface ModelWeights {
  vocab: string[];
  architecture: 'LSTM' | 'SimpleRNN';
  hiddenUnits: number;
  sequenceLength: number;
  // SimpleRNN weights
  W_xh?: number[][]; // input to hidden [hiddenUnits][vocabSize]
  W_hh?: number[][]; // hidden to hidden [hiddenUnits][hiddenUnits]
  b_h?: number[];    // hidden bias [hiddenUnits]
  W_hy?: number[][]; // hidden to output [vocabSize][hiddenUnits]
  b_y?: number[];    // output bias [vocabSize]
  // LSTM weights
  W_f?: number[][]; // forget gate
  W_i?: number[][]; // input gate
  W_c?: number[][]; // cell candidate
  W_o?: number[][]; // output gate
  b_f?: number[];
  b_i?: number[];
  b_c?: number[];
  b_o?: number[];
}

export interface GenerationConfig {
  seed: string;
  length: number;
  temperature: number;
  topK: number;
}

export interface NextCharProbability {
  char: string;
  prob: number;
  logit: number;
}

export interface KaggleCharacterRecord {
  filename: string;
  label: string;
  category: 'digit' | 'uppercase' | 'lowercase';
  classIndex: number;
}
