/**
 * Sample datasets for testing Character-Level RNN Text Generation
 */

export interface SampleCorpus {
  id: string;
  title: string;
  category: string;
  description: string;
  defaultSeed: string;
  text: string;
}

export const SAMPLE_DATASETS: SampleCorpus[] = [
  {
    id: 'handwriting-essay',
    title: 'The Art of Handwriting & The Human Mind',
    category: 'Essay / Literature',
    description: 'A thoughtful essay on handwritten calligraphy, memory, and personal expression. Well-balanced vocabulary.',
    defaultSeed: 'Handwriting is ',
    text: `Handwriting is an intimate mirror of the human spirit. When ink touches paper, thought becomes tangible form. Each curve, stroke, and flourish carries the rhythm of breath and contemplation. Unlike mechanical typography, cursive manuscript reveals the subtle hesitation, speed, and emotion of the writer. Throughout the centuries, scholars kept leather journals filled with ink annotations. Every letter was crafted with intention, blending discipline and creative liberty. To write by hand is to converse slowly with oneself, leaving an enduring trace upon the world.`
  },
  {
    id: 'alice-wonderland',
    title: 'Alice in Wonderland (Classic Excerpt)',
    category: 'Classic Fiction',
    description: 'Lewis Carroll classic text, perfect for studying character sequence learning and English spelling patterns.',
    defaultSeed: 'Alice was beginning ',
    text: `Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, and what is the use of a book, thought Alice without pictures or conversations? So she was considering in her own mind as well as she could, for the hot day made her feel very sleepy and stupid, whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.`
  },
  {
    id: 'ml-notes',
    title: 'Recurrent Neural Networks: Fundamentals',
    category: 'AI & Machine Learning',
    description: 'Technical study notes on sequence modeling, hidden states, backpropagation through time, and temperature sampling.',
    defaultSeed: 'Recurrent neural ',
    text: `Recurrent neural networks are designed to process sequential data by maintaining an internal hidden state. At each time step, the network consumes the current input vector and combines it with the previous hidden state to compute a new activation. In character-level language modeling, the vocabulary consists of discrete symbols including letters, digits, and punctuation. The model learns to predict the conditional probability distribution of the next character given the preceding context. Temperature sampling controls the entropy of predictions during text generation.`
  },
  {
    id: 'shakespeare-sonnet',
    title: 'Shakespearean Sonnets & Soliloquies',
    category: 'Poetry',
    description: 'Elizabethan English with poetic meter, distinct vocabulary, and dramatic cadence.',
    defaultSeed: 'Shall I compare ',
    text: `Shall I compare thee to a summer day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer lease hath all too short a date:
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimmed;
And every fair from fair sometime declines,
By chance or nature changing course untrimmed;
But thy eternal summer shall not fade,
Nor lose possession of that fair thou ow'st.`
  }
];

export const KAGGLE_ENGLISH_CSV_EXPLAINER = {
  datasetName: "English Handwritten Characters Dataset",
  kaggleUrl: "https://www.kaggle.com/datasets/dhruvildave/english-handwritten-characters-dataset",
  csvFile: "english.csv",
  totalImages: 3410,
  classesCount: 62,
  samplesPerClass: 55,
  classBreakdown: [
    { type: "Digits (0-9)", count: 10, samples: "0, 1, 2, 3, 4, 5, 6, 7, 8, 9", imgRange: "img001 to img010" },
    { type: "Uppercase (A-Z)", count: 26, samples: "A, B, C ... X, Y, Z", imgRange: "img011 to img036" },
    { type: "Lowercase (a-z)", count: 26, samples: "a, b, c ... x, y, z", imgRange: "img037 to img062" }
  ],
  structure: [
    { image: "Img/img001-001.png", label: "0" },
    { image: "Img/img010-001.png", label: "9" },
    { image: "Img/img011-001.png", label: "A" },
    { image: "Img/img036-001.png", label: "Z" },
    { image: "Img/img037-001.png", label: "a" },
    { image: "Img/img062-055.png", label: "z" }
  ]
};
