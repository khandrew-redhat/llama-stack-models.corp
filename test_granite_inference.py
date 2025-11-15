#!/usr/bin/env python3

import sys
from llama_stack_client import LlamaStackClient  # pyright: ignore[reportMissingImports]


def test_granite_inference():
    """Test Granite inference provider."""
    
    # Initialize client pointing to local Llama Stack server
    client = LlamaStackClient(
        base_url="http://localhost:8321"
    )
    
    print("=" * 60)
    print("Testing Granite Inference Provider")
    print("=" * 60)
    print()
    
    # Step 1: List available models
    print("Step 1: Listing available models...")
    try:
        # LlamaStackClient.models.list() may return a list or a response object with .data
        response = client.models.list()
        
        # Handle both list and response object with .data attribute
        if hasattr(response, 'data'):
            models = response.data
        elif isinstance(response, list):
            models = response
        else:
            print(f"Error: Unexpected response type: {type(response)}")
            print(f"Response: {response}")
            return False
        
        if not isinstance(models, list):
            print(f"Error: Expected list, got {type(models)}")
            return False
        
        print(f"Found {len(models)} model(s):")
        granite_model_id = None
        
        # Prefer models with correct provider_resource_id format (ibm-granite/...)
        # Avoid models with malformed IDs like "/data/..."
        for model in models:
            # Try both id and identifier (client SDK may use id)
            model_id = getattr(model, 'id', None) or getattr(model, 'identifier', None)
            
            # Extract provider_id and provider_resource_id from identifier if not set
            # Format is: {provider_id}/{provider_resource_id}
            provider_id = getattr(model, 'provider_id', None)
            provider_resource_id = getattr(model, 'provider_resource_id', None) or ''
            
            # If provider_id is None, try to parse from identifier
            if provider_id is None and model_id:
                parts = model_id.split('/', 1)
                if len(parts) == 2:
                    provider_id = parts[0]
                    provider_resource_id = parts[1]
            
            print(f"  - {model_id}")
            print(f"    Provider: {provider_id}, Resource ID: {provider_resource_id}")
            
            # Look for Granite model with valid ID format
            if provider_id == "granite":
                # Prefer models with proper ibm-granite/ prefix in provider_resource_id
                if provider_resource_id and "ibm-granite" in provider_resource_id:
                    if not granite_model_id:  # Use first valid one found
                        granite_model_id = model_id
                        print(f"    Selected Granite model: {model_id}")
                elif not granite_model_id and not provider_resource_id.startswith("/data"):
                    # Fallback to any granite model without /data prefix
                    granite_model_id = model_id
                    print(f"    Found Granite model (fallback): {model_id}")
        
        if not granite_model_id:
            print("\nError: Could not find a valid Granite model in the list!")
            print("Available models:")
            for model in models:
                model_id = getattr(model, 'id', None) or getattr(model, 'identifier', None)
                provider_id = getattr(model, 'provider_id', None)
                print(f"  - {model_id} (provider: {provider_id})")
            return False
        
        print(f"\nUsing model: {granite_model_id}")
        
    except Exception as e:
        print(f"Error listing models: {e}")
        return False
    
    print()
    print("=" * 60)
    
    # Step 2: Send a chat completion request
    print("Step 2: Sending chat completion request...")
    try:
        response = client.chat.completions.create(
            model=granite_model_id,
            messages=[
                {"role": "user", "content": "Hello! What model are you and what can you do?"}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        print("\nChat completion successful!")
        print("\nResponse:")
        print("-" * 60)
        print(response.choices[0].message.content)
        print("-" * 60)
        
        # Print usage information if available
        if hasattr(response, 'usage') and response.usage:
            print(f"\nUsage:")
            print(f"  Prompt tokens: {response.usage.prompt_tokens}")
            print(f"  Completion tokens: {response.usage.completion_tokens}")
            print(f"  Total tokens: {response.usage.total_tokens}")
        
        return True
        
    except Exception as e:
        print(f"\nError sending chat completion: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_messages():
    """Test with a multi-turn conversation."""
    
    client = LlamaStackClient(
        base_url="http://localhost:8321"
    )
    
    print("\n")
    print("=" * 60)
    print("Testing Multi-turn Conversation")
    print("=" * 60)
    
    # First, find the Granite model
    # LlamaStackClient.models.list() may return a list or a response object with .data
    response = client.models.list()
    
    # Handle both list and response object with .data attribute
    if hasattr(response, 'data'):
        models = response.data
    elif isinstance(response, list):
        models = response
    else:
        print(f"Error: Unexpected response type: {type(response)}")
        return False
    
    if not isinstance(models, list):
        print(f"Error: Expected list, got {type(models)}")
        return False
    
    granite_model_id = None
    
    # Prefer models with correct provider_resource_id format
    for model in models:
        model_id = getattr(model, 'id', None) or getattr(model, 'identifier', None)
        provider_id = getattr(model, 'provider_id', None)
        provider_resource_id = getattr(model, 'provider_resource_id', None) or ''
        
        # If provider_id is None, try to parse from identifier
        if provider_id is None and model_id:
            parts = model_id.split('/', 1)
            if len(parts) == 2:
                provider_id = parts[0]
                provider_resource_id = parts[1]
        
        if provider_id == "granite":
            # Prefer models with proper ibm-granite/ prefix
            if provider_resource_id and "ibm-granite" in provider_resource_id:
                granite_model_id = model_id
                break
            elif not granite_model_id and not provider_resource_id.startswith("/data"):
                granite_model_id = model_id
    
    if not granite_model_id:
        print("Error: Could not find Granite model for multi-turn test")
        return False
    
    try:
        # First message
        response1 = client.chat.completions.create(
            model=granite_model_id,
            messages=[
                {"role": "user", "content": "My name is Alice. Remember this."}
            ]
        )
        
        print("\nUser: My name is Alice. Remember this.")
        print(f"Assistant: {response1.choices[0].message.content}")
        
        # Second message with conversation history
        response2 = client.chat.completions.create(
            model=granite_model_id,
            messages=[
                {"role": "user", "content": "My name is Alice. Remember this."},
                {"role": "assistant", "content": response1.choices[0].message.content},
                {"role": "user", "content": "What is my name?"}
            ]
        )
        
        print("\nUser: What is my name?")
        print(f"Assistant: {response2.choices[0].message.content}")
        
        return True
        
    except Exception as e:
        print(f"Error in multi-turn test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n")
    
    # Run basic inference test
    success = test_granite_inference()
    
    if success:
        # Run multi-turn conversation test
        test_multiple_messages()
        
        print("\n")
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n")
        print("=" * 60)
        print("Tests failed!")
        print("=" * 60)
        sys.exit(1)

