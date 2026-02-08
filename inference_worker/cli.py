import logging


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    import uvicorn
    from .web.app import app

    host = "0.0.0.0"
    port = 7861

    logger.info(f"Starting Grid Inference Worker on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
