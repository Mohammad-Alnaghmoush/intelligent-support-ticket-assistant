"""Export the existing frozen head, evidence and vectors for the browser demo."""
import hashlib
import json
import numpy as np
from src.common import ROOT, DATA, ARTIFACTS, CONFIG, BASE_PRIORITY
from src.pipeline import TicketAssistant

def export():
    destination = ROOT / 'web/public/data'
    destination.mkdir(parents=True, exist_ok=True)
    assistant = TicketAssistant(use_llm=False)
    clf = assistant.clf
    manifest = {
        'schema_version': 1,
        'encoder': CONFIG['embedding_model'],
        'revision': CONFIG['embedding_revision'],
        'dtype': 'fp32', 'dimensions': 384, 'max_length': 256,
        'classes': clf.classes_.tolist(),
        'coef': clf.coef_.tolist(), 'intercept': clf.intercept_.tolist(),
        'thresholds': assistant.thresholds,
        'labels': assistant.labels,
        'base_priority': BASE_PRIORITY,
        'max_input_characters': CONFIG['max_input_characters'],
        'history': assistant.retriever.history,
        'kb': assistant.retriever.kb,
        'arabic_replies': json.loads((DATA/'approved_arabic_replies.json').read_text(encoding='utf-8')),
        'source_classifier_sha256': hashlib.sha256((ARTIFACTS/'e5_bilingual.joblib').read_bytes()).hexdigest(),
    }
    for name, vectors in [('history',assistant.retriever.hvectors),('kb',assistant.retriever.kvectors)]:
        target=destination/f'{name}.f32'
        np.asarray(vectors,dtype='<f4').tofile(target)
        manifest[f'{name}_vectors_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
    (destination/'assistant.json').write_text(json.dumps(manifest,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    print('Exported existing head, 1200 historical cases and 33 KB sections. No training performed.')

if __name__ == '__main__':
    export()
