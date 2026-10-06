import { ExpressionType } from '../services/api';

export interface SampleImageItem {
  id: string;
  image: string;
  expectedEmotion: ExpressionType;
  title: string;
  description: string;
  attribution: string;
}

export const SAMPLE_IMAGES: SampleImageItem[] = [
  // --- NEUTRAL (3) ---
  {
    id: 'neutral-01',
    image: '/samples/neutral/neutral_01.jpg',
    expectedEmotion: 'Neutral',
    title: 'Neutral 01',
    description: 'Calm frontal facial expression with relaxed jaw and level gaze',
    attribution: 'Evaluation Dataset (Standard Reference)',
  },
  {
    id: 'neutral-02',
    image: '/samples/neutral/neutral_02.jpg',
    expectedEmotion: 'Neutral',
    title: 'Neutral 02',
    description: 'Resting neutral face with balanced facial muscle tone',
    attribution: 'Frontal Evaluation Reference',
  },
  {
    id: 'neutral-03',
    image: '/samples/neutral/neutral_03.jpg',
    expectedEmotion: 'Neutral',
    title: 'Neutral 03',
    description: 'Even frontal posture with relaxed lips and steady neutral focus',
    attribution: 'Documentary Evaluation Reference',
  },

  // --- HAPPY (3) ---
  {
    id: 'happy-01',
    image: '/samples/happy/happy_01.jpg',
    expectedEmotion: 'Happy',
    title: 'Happy 01',
    description: 'Genuine Duchenne smile with cheek elevation and eye crinkle',
    attribution: 'Frontal Expression Sample',
  },
  {
    id: 'happy-02',
    image: '/samples/happy/happy_02.jpg',
    expectedEmotion: 'Happy',
    title: 'Happy 02',
    description: 'Broad joyous smile with visible teeth and elevated zygomaticus',
    attribution: 'Studio Evaluation Sample',
  },
  {
    id: 'happy-03',
    image: '/samples/happy/happy_03.jpg',
    expectedEmotion: 'Happy',
    title: 'Happy 03',
    description: 'Radiant smiling expression with warm relaxed facial lighting',
    attribution: 'Frontal Portrait Sample',
  },

  // --- SAD (3) ---
  {
    id: 'sad-01',
    image: '/samples/sad/sad_01.jpg',
    expectedEmotion: 'Sad',
    title: 'Sad 01',
    description: 'Downturned lip corners with subtle inner eyebrow elevation',
    attribution: 'Frontal Expression Sample',
  },
  {
    id: 'sad-02',
    image: '/samples/sad/sad_02.jpg',
    expectedEmotion: 'Sad',
    title: 'Sad 02',
    description: 'Melancholic expression with lowered gaze and furrowed forehead',
    attribution: 'Studio Evaluation Sample',
  },
  {
    id: 'sad-03',
    image: '/samples/sad/sad_03.jpg',
    expectedEmotion: 'Sad',
    title: 'Sad 03',
    description: 'Somber sorrowful expression with depressed lip corners',
    attribution: 'Documentary Evaluation Reference',
  },

  // --- SURPRISE (3) ---
  {
    id: 'surprise-01',
    image: '/samples/surprise/surprise_01.jpg',
    expectedEmotion: 'Surprise',
    title: 'Surprise 01',
    description: 'Widened ocular aperture with elevated frontalis arch',
    attribution: 'Frontal Expression Sample',
  },
  {
    id: 'surprise-02',
    image: '/samples/surprise/surprise_02.jpg',
    expectedEmotion: 'Surprise',
    title: 'Surprise 02',
    description: 'Astonished expression with open mouth and raised eyebrows',
    attribution: 'Studio Evaluation Sample',
  },
  {
    id: 'surprise-03',
    image: '/samples/surprise/surprise_03.jpg',
    expectedEmotion: 'Surprise',
    title: 'Surprise 03',
    description: 'Startled expression with widened eyes and jaw drop',
    attribution: 'Frontal Portrait Sample',
  },

  // --- FEAR (3) ---
  {
    id: 'fear-01',
    image: '/samples/fear/fear_01.jpg',
    expectedEmotion: 'Fear',
    title: 'Fear 01',
    description: 'Wide open eyes with upper eyelid retraction and tense lips',
    attribution: 'Evaluation Dataset Asset',
  },
  {
    id: 'fear-02',
    image: '/samples/fear/fear_02.jpg',
    expectedEmotion: 'Fear',
    title: 'Fear 02',
    description: 'Alarmed facial posture with pulled-back mouth corners and raised brows',
    attribution: 'Evaluation Dataset Asset',
  },
  {
    id: 'fear-03',
    image: '/samples/fear/fear_03.jpg',
    expectedEmotion: 'Fear',
    title: 'Fear 03',
    description: 'Apprehensive expression with tense neck and widened sclera',
    attribution: 'Evaluation Dataset Asset',
  },

  // --- DISGUST (2) ---
  {
    id: 'disgust-01',
    image: '/samples/disgust/disgust_01.jpg',
    expectedEmotion: 'Disgust',
    title: 'Disgust 01',
    description: 'Wrinkled nose bridge with raised upper lip and narrowed eyes',
    attribution: 'Evaluation Dataset Asset',
  },
  {
    id: 'disgust-02',
    image: '/samples/disgust/disgust_02.jpg',
    expectedEmotion: 'Disgust',
    title: 'Disgust 02',
    description: 'Aversive facial grimace with elevated nasolabial fold',
    attribution: 'Evaluation Dataset Asset',
  },

  // --- ANGRY (3) ---
  {
    id: 'angry-01',
    image: '/samples/angry/angry_01.jpg',
    expectedEmotion: 'Angry',
    title: 'Angry 01',
    description: 'Lowered knitted eyebrows with intense direct glare and tightened lips',
    attribution: 'Evaluation Dataset Asset',
  },
  {
    id: 'angry-02',
    image: '/samples/angry/angry_02.jpg',
    expectedEmotion: 'Angry',
    title: 'Angry 02',
    description: 'Stern brow furrow with compressed jaw and tense mouth line',
    attribution: 'Evaluation Dataset Asset',
  },
  {
    id: 'angry-03',
    image: '/samples/angry/angry_03.jpg',
    expectedEmotion: 'Angry',
    title: 'Angry 03',
    description: 'Fierce angry expression with contracted corrugator muscles',
    attribution: 'Evaluation Dataset Asset',
  },
];
