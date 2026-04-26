import time
import asyncio
import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..services.inference_service import get_inference_service
from ..services.metrics_utils import (
    compute_per_class_coverage,
    compute_iou_per_class,
    create_segmentation_overlay,
    create_side_by_side_comparison,
    format_results_for_json,
    predictions_to_image
)
from ..services.input_handler import InputHandler


router = APIRouter(tags=['inference'])


class InferenceRequest(BaseModel):
    """Request model for image inference."""
    image_base64: str
    return_overlay: bool = True
    return_comparison: bool = False


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    architecture: str
    backbone: str
    num_classes: int
    input_height: int
    input_width: int
    device: str
    classes: list


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get information about the loaded model.
    """
    try:
        service = get_inference_service()
        info = service.get_model_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/infer")
async def infer_single_image(request: InferenceRequest):
    """
    Run inference on a single image provided as base64.
    
    Returns:
    - predictions: Predicted class map as base64 image
    - coverage: Per-class coverage percentages
    - overlay: Blended image if return_overlay=True
    - comparison: Side-by-side comparison if return_comparison=True
    - metrics: Inference time and other stats
    """
    try:
        service = get_inference_service()
        config = service.get_config()
        
        # Load image from base64
        start_time = time.time()
        image = InputHandler.load_image_from_base64(request.image_base64)
        
        # Run inference
        predictions, _ = service.infer(image)
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Compute metrics
        coverage = compute_per_class_coverage(predictions)
        class_ious, mean_iou = compute_iou_per_class(predictions, ground_truth=None)
        
        # Prepare response
        response = {
            'inference_time_ms': inference_time_ms,
            'predictions_shape': predictions.shape,
            'metrics': format_results_for_json(
                predictions, coverage, class_ious, mean_iou,
                config['classes'], inference_time_ms
            ),
            'coverage': coverage,
            'image_base64': InputHandler.image_to_base64(predictions_to_image(predictions))
        }
        
        # Add overlay if requested
        if request.return_overlay:
            overlay = create_segmentation_overlay(image, predictions)
            response['overlay_base64'] = InputHandler.image_to_base64(overlay)
        
        # Add comparison if requested
        if request.return_comparison:
            comparison = create_side_by_side_comparison(image, predictions)
            response['comparison_base64'] = InputHandler.image_to_base64(comparison)
        
        return JSONResponse(response)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/infer-file")
async def infer_from_file(file: UploadFile = File(...)):
    """
    Run inference on an uploaded image file.
    
    Returns:
    - predictions: Predicted class map
    - coverage: Per-class coverage
    - overlay: Blended image
    - metrics: Performance stats
    """
    try:
        service = get_inference_service()
        config = service.get_config()
        
        # Read uploaded file
        contents = await file.read()
        image = InputHandler.load_image_from_bytes(contents)
        
        # Run inference
        start_time = time.time()
        predictions, _ = service.infer(image)
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Compute metrics
        coverage = compute_per_class_coverage(predictions)
        class_ious, mean_iou = compute_iou_per_class(predictions)
        
        # Create overlay
        overlay = create_segmentation_overlay(image, predictions)
        comparison = create_side_by_side_comparison(image, predictions)
        
        response = {
            'filename': file.filename,
            'inference_time_ms': inference_time_ms,
            'predictions_shape': predictions.shape,
            'metrics': format_results_for_json(
                predictions, coverage, class_ious, mean_iou,
                config['classes'], inference_time_ms
            ),
            'coverage': coverage,
            'image_base64': InputHandler.image_to_base64(predictions_to_image(predictions)),
            'overlay_base64': InputHandler.image_to_base64(overlay),
            'comparison_base64': InputHandler.image_to_base64(comparison)
        }
        
        return JSONResponse(response)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.websocket("/ws/infer")
async def websocket_infer(websocket: WebSocket):
    """
    WebSocket endpoint for streaming frame inference.
    
    Expected messages:
    - Frame: {"type": "frame", "image_base64": "..."}
    - Info: {"type": "info"} -> Returns model info
    - Stop: {"type": "stop"}
    
    Responses:
    - {"type": "results", "predictions_base64": "...", "metrics": {...}}
    - {"type": "error", "message": "..."}
    - {"type": "info", "data": {...}}
    """
    await websocket.accept()
    
    try:
        service = get_inference_service()
        config = service.get_config()
        
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            msg_type = data.get('type', 'frame')
            
            if msg_type == 'stop':
                break
            
            elif msg_type == 'info':
                info = service.get_model_info()
                await websocket.send_json({
                    'type': 'info',
                    'data': info
                })
            
            elif msg_type == 'frame':
                try:
                    # Decode image
                    image = InputHandler.load_image_from_base64(data['image_base64'])
                    
                    # Run inference
                    start_time = time.time()
                    predictions, _ = service.infer(image)
                    inference_time_ms = (time.time() - start_time) * 1000
                    
                    # Compute metrics
                    coverage = compute_per_class_coverage(predictions)
                    class_ious, mean_iou = compute_iou_per_class(predictions)
                    
                    # Create visualizations
                    pred_image = predictions_to_image(predictions)
                    overlay = create_segmentation_overlay(image, predictions, alpha=0.7)
                    
                    # Send results
                    await websocket.send_json({
                        'type': 'results',
                        'predictions_base64': InputHandler.image_to_base64(pred_image),
                        'overlay_base64': InputHandler.image_to_base64(overlay),
                        'inference_time_ms': inference_time_ms,
                        'coverage': coverage,
                        'metrics': {
                            'mean_iou': float(mean_iou) if not np.isnan(mean_iou) else None,
                            'per_class_iou': {
                                config['classes'][i]: (float(v) if not np.isnan(v) else None)
                                for i, v in class_ious.items()
                            }
                        }
                    })
                
                except Exception as e:
                    await websocket.send_json({
                        'type': 'error',
                        'message': str(e)
                    })
    
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_json({'type': 'error', 'message': str(e)})
        except:
            pass
