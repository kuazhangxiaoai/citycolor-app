import uvicorn

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import torch
from starlette.middleware.cors import CORSMiddleware

from ultralytics.utils import YAML



def main():
    print("CityColor app start")
    print("CityColor app end")
    return

if __name__ == '__main__':
    main()