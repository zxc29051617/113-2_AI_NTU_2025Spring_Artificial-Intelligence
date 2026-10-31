"""Audio judges: CLAP text-music similarity and a GTZAN music-genre classifier."""
import librosa
import numpy as np
import torch
import torch.nn.functional as F

CLAP_ORIGINAL = "laion/clap-htsat-unfused"  # LAION-CLAP (Wu et al. 2023), used to report CLAP scores
CLAP_SELECT = "laion/clap-htsat-fused"  # another LAION-CLAP checkpoint, used only to select candidates
GENRE_MODEL = "dima806/music_genres_classification"  # wav2vec2-base fine-tuned on GTZAN
GTZAN = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]


class ClapJudge:
    def __init__(self, name=CLAP_ORIGINAL, device="cuda"):
        from transformers import ClapModel, ClapProcessor
        self.device = device
        self.model = ClapModel.from_pretrained(name).to(device).eval()
        self.proc = ClapProcessor.from_pretrained(name)

    @torch.no_grad()
    def text_emb(self, texts):
        inp = self.proc(text=list(texts), return_tensors="pt", padding=True).to(self.device)
        return F.normalize(self.model.get_text_features(**inp), dim=-1).cpu().numpy()

    @torch.no_grad()
    def audio_emb(self, wavs, sr, batch=8):
        out = []
        for i in range(0, len(wavs), batch):
            chunk = [librosa.resample(np.asarray(w, dtype=np.float32), orig_sr=sr, target_sr=48000)
                     for w in wavs[i:i + batch]]
            inp = self.proc(audios=chunk, sampling_rate=48000, return_tensors="pt").to(self.device)
            out.append(F.normalize(self.model.get_audio_features(**inp), dim=-1).cpu().numpy())
        return np.concatenate(out)

    @staticmethod
    def score(text_e, audio_e):
        """CLAP score of the original report: max(0, cos(text, music)), matrix [text, music]."""
        return np.maximum(0.0, text_e @ audio_e.T)


class GenreJudge:
    def __init__(self, name=GENRE_MODEL, device="cuda"):
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        self.device = device
        self.fe = AutoFeatureExtractor.from_pretrained(name)
        self.model = AutoModelForAudioClassification.from_pretrained(name).to(device).eval()
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    @torch.no_grad()
    def probs(self, wavs, sr, batch=8):
        """Class probabilities, columns ordered as GTZAN (alphabetical)."""
        out = []
        for i in range(0, len(wavs), batch):
            chunk = [librosa.resample(np.asarray(w, dtype=np.float32), orig_sr=sr, target_sr=16000)
                     for w in wavs[i:i + batch]]
            inp = self.fe(chunk, sampling_rate=16000, return_tensors="pt", padding=True).to(self.device)
            p = torch.softmax(self.model(**inp).logits, dim=-1).cpu().numpy()
            out.append(p[:, [self.labels.index(g) for g in GTZAN]])
        return np.concatenate(out)
