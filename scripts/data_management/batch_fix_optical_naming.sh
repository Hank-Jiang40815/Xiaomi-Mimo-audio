#!/bin/bash
#
# 批次修正 optical 資料集的檔案命名
# 處理 mix/ 和 spk1/ 兩個目錄
#

echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║         批次修正 Optical 資料集檔案命名                                  ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo ""

# 檢查是否為 dry-run 模式
DRY_RUN_FLAG=""
if [ "$1" == "--dry-run" ]; then
    DRY_RUN_FLAG="--dry-run"
    echo "⚠️  預覽模式 (不會實際重命名)"
else
    echo "✅ 執行模式 (將實際重命名檔案)"
    echo ""
    read -p "確定要重命名檔案嗎？(y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消。"
        exit 1
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "處理 mix/ 目錄"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python scripts/data_management/fix_optical_naming.py \
    --dir examples/optical/mix \
    $DRY_RUN_FLAG

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "處理 spk1/ 目錄"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python scripts/data_management/fix_optical_naming.py \
    --dir examples/optical/spk1 \
    $DRY_RUN_FLAG

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║                           完成！                                          ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"

if [ "$1" == "--dry-run" ]; then
    echo ""
    echo "💡 提示: 執行不加 --dry-run 來實際重命名檔案"
    echo "   bash scripts/data_management/batch_fix_optical_naming.sh"
fi
