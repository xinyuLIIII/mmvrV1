#!/usr/bin/env bash
# 重新生成交接包：handoff/mmvrV1_local_changes.patch + handoff/MANIFEST.txt，
# 并刷新 handoff/HANDOFF_FOR_AI.md 和 handoff/PROMPT_LINUX.md（发给新机器 AI 的自包含提示词）
# 里的「版本」块（<!-- VERSION:BEGIN/END --> 之间）。
#
# 对仓库只读：不 commit / 不改真实 index / 不改工作区文件，只写 handoff/。
# 用临时 GIT_INDEX_FILE 把「相对 HEAD 的 tracked 修改 + 未跟踪文件（排除下列路径）」
# 加进去，再 git diff --cached --binary HEAD 生成补丁。
#
#   bash scripts/make_handoff.sh
#
# 排除：.merge_backup/ data/ handoff/ experiments/ logs/ __pycache__/ *.pyc
#       *.pdf *.docx wangchen_extract.txt（.gitignore 忽略的文件本来就不会进来）
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
OUT_DIR=handoff
PATCH=$OUT_DIR/mmvrV1_local_changes.patch
MANIFEST=$OUT_DIR/MANIFEST.txt
DOCS=("$OUT_DIR/HANDOFF_FOR_AI.md" "$OUT_DIR/PROMPT_LINUX.md")
mkdir -p "$OUT_DIR"

is_excluded() {
    case "$1" in
        .merge_backup/*|data/*|handoff/*|experiments/*|logs/*) return 0 ;;
        */__pycache__/*|__pycache__/*|*.pyc) return 0 ;;
        *.pdf|*.docx|wangchen_extract.txt) return 0 ;;
    esac
    return 1
}

# 候选：tracked 中相对 HEAD 有改动/删除的 + 未跟踪且未被 .gitignore 忽略的
PATHS=()
while IFS= read -r -d '' p; do
    is_excluded "$p" || PATHS+=("$p")
done < <({ git -c core.quotepath=false diff -z --name-only HEAD;
           git -c core.quotepath=false ls-files -z --others --exclude-standard; } | sort -zu)

if [ ${#PATHS[@]} -eq 0 ]; then
    echo "no local changes to package" >&2
    exit 1
fi

BASE=$(git rev-parse HEAD)
TMP_INDEX=$(mktemp)
CHECK_INDEX=$(mktemp)
trap 'rm -f "$TMP_INDEX" "$CHECK_INDEX"' EXIT

GIT_INDEX_FILE=$TMP_INDEX git read-tree HEAD
# 行尾统一成 LF 入补丁：HEAD 里的 config.py 是 CRLF/LF 混合，普通 add 会因
# 「index 里已有 CR」而保留 CRLF，所以再对 tracked 文件做一次 --renormalize。
GIT_INDEX_FILE=$TMP_INDEX git -c core.autocrlf=true add -A -- "${PATHS[@]}" 2>/dev/null
GIT_INDEX_FILE=$TMP_INDEX git -c core.autocrlf=true add --renormalize -- "${PATHS[@]}" 2>/dev/null
GIT_INDEX_FILE=$TMP_INDEX git diff --cached --binary --full-index HEAD > "$PATCH"
STAT=$(GIT_INDEX_FILE=$TMP_INDEX git -c core.quotepath=false diff --cached --stat=200 HEAD)
NAMES=$(GIT_INDEX_FILE=$TMP_INDEX git -c core.quotepath=false diff --cached --name-status HEAD)

# 自检：在另一份以 HEAD 为基准的临时 index 上 check 补丁
GIT_INDEX_FILE=$CHECK_INDEX git read-tree HEAD
GIT_INDEX_FILE=$CHECK_INDEX git apply --cached --check "$PATCH"

N_FILES=$(printf '%s\n' "$NAMES" | grep -c .)
SIZE=$(wc -c < "$PATCH" | tr -d ' ')
SHA=$(sha256sum "$PATCH" | cut -d' ' -f1)
BUILT=$(date '+%Y-%m-%d %H:%M:%S %z')

{
    echo "base_commit: $BASE"
    echo "built_at:    $BUILT"
    echo "patch:       $PATCH"
    echo "bytes:       $SIZE"
    echo "sha256:      $SHA"
    echo "files:       $N_FILES"
    echo
    echo "$NAMES"
    echo
    echo "$STAT"
} > "$MANIFEST"

BLOCK=$(mktemp)
{
    echo '| 项 | 值 |'
    echo '|---|---|'
    echo "| 基线提交 | \`$BASE\` |"
    echo "| 补丁生成时间 | $BUILT |"
    echo "| 补丁文件 | \`mmvrV1_local_changes.patch\`（$SIZE 字节，$N_FILES 个文件） |"
    echo "| sha256 | \`$SHA\` |"
    echo
    echo '文件列表（A=新增，M=修改）：'
    echo
    echo '```'
    echo "$NAMES"
    echo '```'
} > "$BLOCK"
for DOC in "${DOCS[@]}"; do
    [ -f "$DOC" ] && grep -q '<!-- VERSION:BEGIN -->' "$DOC" || continue
    awk -v blk="$BLOCK" '
        /<!-- VERSION:BEGIN -->/ { print; while ((getline l < blk) > 0) print l; skip = 1; next }
        /<!-- VERSION:END -->/   { skip = 0 }
        !skip { print }
    ' "$DOC" > "$DOC.tmp" && mv "$DOC.tmp" "$DOC"
    echo "updated version block in $DOC"
done
rm -f "$BLOCK"

cat "$MANIFEST"
