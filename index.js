import { pipeline } from "@xenova/transformers";
import wavefile from "wavefile";
import fs from "fs";

const embed =
  "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/speaker_embeddings.bin";
const phrase = "don't give up, keep figthing";

const synthesizer = await pipeline("text-to-speech", "Xenova/speecht5_tts", {
  quantized: false,
});

const output = await synthesizer(phrase, { speaker_embeddings: embed });

const wav = new wavefile.WaveFile();
wav.fromScratch(1, output.sampling_rate, '32f', output.audio);
fs.writeFileSync('out.wav', wav.toBuffer());
