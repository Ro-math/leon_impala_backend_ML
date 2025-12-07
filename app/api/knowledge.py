from fastapi import APIRouter, HTTPException
from app.models.requests import KnowledgeSaveRequest, KnowledgeLoadRequest
from app.models.responses import KnowledgeResponse, KnowledgeFilesResponse, KnowledgeQueryResponse
from app.api.training import training_manager

router = APIRouter()

@router.get("/base")
def get_knowledge_base():
    # Return the full KB content for inspection
    return {
        "q_table_size": len(training_manager.kb.q_table),
        "abstractions_count": len(training_manager.kb.abstractions),
        "q_table": training_manager.kb.q_table,
        "abstractions": training_manager.kb.abstractions
    }

@router.get("/download")
def download_knowledge():
    from fastapi.responses import FileResponse
    import os
    
    # Ensure final file exists
    filepath = "data/knowledge/knowledge_final.json"
    if not os.path.exists(filepath):
        # Try checkpoint
        filepath = "data/knowledge/knowledge_checkpoint.json"
        
    if not os.path.exists(filepath):
        # Save current state to temp
        training_manager.kb.save("knowledge_download")
        filepath = "data/knowledge/knowledge_download.json"
        
    return FileResponse(filepath, media_type='application/json', filename="knowledge_base.json")

@router.get("/abstractions")
def get_abstractions():
    return training_manager.kb.abstractions

@router.post("/save")
def save_knowledge(request: KnowledgeSaveRequest):
    training_manager.kb.save(request.filename, request.format)
    return {"message": "Knowledge saved"}

@router.post("/load")
def load_knowledge(request: KnowledgeLoadRequest):
    training_manager.kb.load(request.filename)
    return {"message": "Knowledge loaded"}

@router.delete("/clear")
def clear_knowledge():
    training_manager.kb.clear()
    return {"message": "Knowledge cleared"}

@router.post("/reset")
def reset_learning():
    """Reset all learning data, statistics, and delete knowledge files"""
    import os
    import shutil
    
    # Reset all learning data and statistics
    training_manager.reset_learning()
    
    # Delete all knowledge files
    knowledge_dir = "data/knowledge"
    deleted_files = []
    
    if os.path.exists(knowledge_dir):
        for filename in os.listdir(knowledge_dir):
            filepath = os.path.join(knowledge_dir, filename)
            if os.path.isfile(filepath):
                os.remove(filepath)
                deleted_files.append(filename)
    
    return {
        "message": "Learning data reset successfully",
        "details": {
            "knowledge_base_cleared": True,
            "statistics_reset": True,
            "files_deleted": deleted_files,
            "files_deleted_count": len(deleted_files)
        }
    }

@router.get("/files", response_model=KnowledgeFilesResponse)
def list_knowledge_files():
    import os
    files = []
    knowledge_dir = "data/knowledge"
    if os.path.exists(knowledge_dir):
        files = os.listdir(knowledge_dir)
    return KnowledgeFilesResponse(files=files)

@router.get("/query", response_model=KnowledgeQueryResponse)
def query_knowledge(lion_position: int, impala_action: str):
    # Validate inputs
    from app.core.entities import GameMap, ImpalaAction, LionState
    
    if lion_position not in GameMap.valid_lion_positions:
        raise HTTPException(status_code=400, detail="Invalid lion position")
        
    try:
        imp_act = ImpalaAction(impala_action)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid impala action")
        
    lion_pos = GameMap.valid_lion_positions[lion_position]
    
    # We query for ALL Lion States? Or just Normal?
    # Prompt says "decisiones aprendidas para ese estado".
    # State includes LionState.
    # Let's return data for LionState.NORMAL as default or aggregate?
    # Or maybe we take LionState as param?
    # Prompt only says "lion_position" and "impala_action".
    # I'll assume LionState.NORMAL for the query or return all states?
    # Let's return for NORMAL.
    
    state_key = training_manager.agent.get_state_key(lion_pos, imp_act, LionState.NORMAL)
    q_values = training_manager.kb.q_table.get(state_key, {})
    
    best_action = "unknown"
    if q_values:
        best_action = max(q_values, key=q_values.get)
        
    # Find matching rules
    rules = []
    for rule in training_manager.kb.abstractions:
        if f"Lion at {lion_pos[0]},{lion_pos[1]}" in rule and impala_action in rule:
            rules.append(rule)
            
    return KnowledgeQueryResponse(
        best_action=best_action,
        q_values=q_values,
        matching_rules=rules
    )
