# Server-hosted application

The existing static browser demo downloads the encoder onto each visitor's device. To avoid this download, run the original Python application on a server. Visitors download the interface; messages are sent to the server for processing. Model weights stay on the server. This changes the execution location, not the classifier, thresholds, retrieval method or reply rules.

The Dockerfile runs the existing dark Gradio interface and default `use_llm=False` pipeline. It caches the original encoder and retrieval vectors during the image build. It does not retrain, quantize or lower confidence thresholds. This deployment is independent of the existing static Site; adding these files does not change that Site's execution mode.

## Temporary Colab link

After completing the notebook setup, run this cell from the project root. Close a previously launched interface first if its port is occupied.

```python
from src.pipeline import TicketAssistant
from app import build_app

assistant = TicketAssistant(use_llm=False)
demo = build_app(assistant)
demo.queue(max_size=20).launch(share=True)
```

Share the public `https://...gradio.live` URL printed by Gradio, not localhost. Visitors do not download the model. The Colab machine downloads and runs it. The link is temporary and requires the Colab runtime and interface to remain active. Do not use real customer information in a public course demo.

## Container deployment

With Docker installed, run from the repository root:

```bash
docker build -t support-ticket-assistant .
docker run --rm -p 7860:7860 support-ticket-assistant
```

Open `http://localhost:7860`. Image building downloads Python dependencies and the original model, so it needs network access and disk space. The final image includes the model cache. The application listens on `0.0.0.0` and uses the hosting provider's `PORT` environment variable, falling back to 7860.

## Railway deployment

1. Sign in to Railway and connect the GitHub repository `Mohammad-Alnaghmoush/intelligent-support-ticket-assistant`.
2. Create a service from the repository's `main` branch with the repository root as the service root.
3. Railway detects the root Dockerfile. Use its default Docker build and start command; do not set the root to `web/`.
4. Provide sufficient memory for PyTorch and the encoder. Start with at least 2 GB and check actual memory usage; this is a starting configuration, not a measured capacity guarantee.
5. Wait for the image build and application startup. In the service networking settings, generate a public domain for the application port.
6. Test the English and Arabic examples through that domain, then share it as the server-hosted demo link.

Railway is an external hosting service with account, plan and usage charges. Review the account's available credits, limits and pricing before deploying. No Railway credentials or payment details belong in this repository.

The source configuration is prepared, but a cloud service still needs to be created in an authorized hosting account. The container image was not built in the authoring environment because a Docker engine was unavailable. The underlying Python application and model parity were tested separately; this is not a claim that a new hosted server is already running.

References: https://docs.railway.com/builds/dockerfiles and https://docs.railway.com/pricing/plans
