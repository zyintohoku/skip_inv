#!/bin/bash
# Move *ori.png and *edi.png from aidi_gs* to outputs/editing/aidi_gs*/

PROJECT_DIR=${PROJECT_DIR:-$PWD}

EDITING_DIR="outputs/editing"
SOURCE_DIRS=(aidi_gs1 aidi_gs3 aidi_gs5 aidi_gs7)

echo "Moving editing images to $EDITING_DIR"
echo "="*60

for dir in "${SOURCE_DIRS[@]}"; do
    source_path="outputs/$dir"
    dest_path="$EDITING_DIR/$dir"
    
    if [ ! -d "$source_path" ]; then
        echo "⚠ Skipping $dir: source directory not found"
        continue
    fi
    
    # Create destination directory
    mkdir -p "$dest_path"
    
    # Count files to move
    ori_count=$(ls "$source_path"/*ori.png 2>/dev/null | wc -l)
    edi_count=$(ls "$source_path"/*edi.png 2>/dev/null | wc -l)
    total=$((ori_count + edi_count))
    
    if [ $total -eq 0 ]; then
        echo "⚠ $dir: No *ori.png or *edi.png files found"
        continue
    fi
    
    echo "📁 $dir: Moving $ori_count ori.png + $edi_count edi.png files..."
    
    # Move files
    mv "$source_path"/*ori.png "$dest_path/" 2>/dev/null
    mv "$source_path"/*edi.png "$dest_path/" 2>/dev/null
    
    # Verify
    moved_count=$(ls "$dest_path"/*.png 2>/dev/null | wc -l)
    echo "   ✓ Moved $moved_count files to $dest_path"
done

echo ""
echo "✨ Done! Editing images moved to $EDITING_DIR/"
echo ""
echo "Summary:"
for dir in "${SOURCE_DIRS[@]}"; do
    dest_path="$EDITING_DIR/$dir"
    if [ -d "$dest_path" ]; then
        count=$(ls "$dest_path"/*.png 2>/dev/null | wc -l)
        echo "  $dir: $count images"
    fi
done
