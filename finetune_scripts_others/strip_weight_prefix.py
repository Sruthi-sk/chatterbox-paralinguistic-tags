import torch
from safetensors.torch import load_file, save_file
from pathlib import Path
import argparse

def strip_prefix_and_resave(checkpoint_path: str, prefix_to_strip: str, output_path: str):
    """
    Loads a PyTorch model checkpoint, strips a prefix from its state_dict keys,
    and saves the modified state_dict to a new .safetensors file.

    Args:
        checkpoint_path (str): Path to the input checkpoint file (.bin or .safetensors).
        prefix_to_strip (str): The prefix to remove from the keys (e.g., "t3.").
        output_path (str): Path to save the new .safetensors file.
    """
    ckpt_path = Path(checkpoint_path)
    out_path = Path(output_path)

    if not ckpt_path.exists():
        print(f"Error: Checkpoint file not found at {ckpt_path}")
        return

    if out_path.suffix != ".safetensors":
        print(f"Warning: Output path '{out_path}' does not have a .safetensors extension. Saving as .safetensors anyway.")
        out_path = out_path.with_suffix(".safetensors")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from: {ckpt_path}")
    if ckpt_path.suffix == ".safetensors":
        try:
            state_dict = load_file(ckpt_path, device="cpu")
        except Exception as e:
            print(f"Error loading .safetensors file: {e}")
            # Try loading as a full model state dict if it's a PEFT adapter or similar
            try:
                print("Attempting to load as a full model state dict which might contain adapter weights...")
                # This might be needed if the safetensors file is not just a flat state_dict
                # but a more complex structure (less common for raw weights).
                # For simple state_dict saves, load_file is usually enough.
                # If it's an adapter, the keys might already be unprefixed.
                data = torch.load(ckpt_path, map_location="cpu")
                if "state_dict" in data:
                    state_dict = data["state_dict"]
                elif isinstance(data, dict): # If it's already a state_dict
                    state_dict = data
                else:
                    raise ValueError("Unsupported .safetensors structure for this script.")
            except Exception as e_alt:
                print(f"Alternative loading failed: {e_alt}")
                return
    elif ckpt_path.suffix == ".bin":
        try:
            data = torch.load(ckpt_path, map_location="cpu")
            # .bin files from Trainer might be the model itself or a dict containing state_dict
            if isinstance(data, dict) and "state_dict" in data: # Common for Lightning checkpoints
                state_dict = data["state_dict"]
            elif isinstance(data, dict) and not any(k.startswith("optimizer") or k.startswith("lr_scheduler") for k in data.keys()):
                # Likely a raw state_dict (common for HF Trainer saves)
                state_dict = data
            elif hasattr(data, 'state_dict'): # If it's a model instance
                 state_dict = data.state_dict()
            else:
                print(f"Error: pytorch_model.bin does not seem to be a raw state_dict or contain a 'state_dict' key.")
                print(f"         Content keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                return
        except Exception as e:
            print(f"Error loading .bin file: {e}")
            return
    else:
        print(f"Error: Unsupported file extension '{ckpt_path.suffix}'. Please use .bin or .safetensors.")
        return

    if not isinstance(state_dict, dict):
        print(f"Error: Loaded data is not a state_dict (dictionary). Type: {type(state_dict)}")
        return

    print(f"Original number of keys: {len(state_dict)}")
    # print(f"Original keys sample: {list(state_dict.keys())[:5]}")

    new_state_dict = {}
    stripped_count = 0
    kept_count = 0

    for key, value in state_dict.items():
        if key.startswith(prefix_to_strip):
            new_key = key[len(prefix_to_strip):]
            new_state_dict[new_key] = value
            stripped_count += 1
        else:
            # If you only want the stripped part, you might not want to keep these.
            # For creating a state_dict *only* for the sub-module, you'd typically ignore these.
            # If the input checkpoint *only* contains keys for the sub-module (all prefixed),
            # then this 'else' branch might not be hit much.
            new_state_dict[key] = value
            kept_count += 1 
            if kept_count < 5 : # Log a few unstripped keys
                 print(f"  Keeping unstripped key (or key did not match prefix): {key}")


    if stripped_count == 0:
        print(f"Warning: No keys found with the prefix '{prefix_to_strip}'. The new state_dict will be empty or identical to the input if unstripped keys were kept.")
        if not new_state_dict and kept_count > 0: # only unstripped keys were kept by uncommenting
             print("Saving all original keys as no prefix matched.")
             new_state_dict = state_dict # Fallback to save original if no stripping happened and unstripped were not kept
        elif not new_state_dict and kept_count == 0:
             print("Resulting state_dict is empty. Aborting save.")
             return


    print(f"Number of keys after stripping prefix '{prefix_to_strip}': {len(new_state_dict)}")
    print(f"  {stripped_count} keys had prefix stripped.")
    print(f"  {kept_count} keys did not have the prefix (or were kept if uncommented).")
    # print(f"New keys sample: {list(new_state_dict.keys())[:5]}")

    if not new_state_dict and stripped_count > 0 : # Should not happen if stripped_count > 0
        print("Error: Stripped keys but new_state_dict is empty. This is a bug.")
        return
    if not new_state_dict and kept_count == 0 and stripped_count == 0:
        print("No keys to save. Aborting.")
        return


    try:
        save_file(new_state_dict, out_path)
        print(f"Successfully stripped prefix and saved new checkpoint to: {out_path}")
    except Exception as e:
        print(f"Error saving .safetensors file: {e}")



strip_prefix_and_resave("path/to/checkpoint.safetensors", "t3.", "path/to/store/checkpoint.safetensors")