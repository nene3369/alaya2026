"""Lightweight internationalization (i18n) for LMM messages.

Supports: ja (default), en, zh, ko, es, hi, fr.

Usage:
    from lmm.i18n import t, set_locale

    set_locale("en")
    print(t("solver.no_quantum"))  # "No quantum computer required"
"""

from __future__ import annotations

import os

_LOCALE = os.environ.get("LMM_LANG", "ja")

_MESSAGES: dict[str, dict[str, str]] = {
    # -- Core solver messages --
    "solver.no_quantum": {
        "ja": "量子コンピュータ不要",
        "en": "No quantum computer required",
        "zh": "无需量子计算机",
        "ko": "양자 컴퓨터 불필요",
        "es": "No se requiere computadora cuántica",
        "hi": "क्वांटम कंप्यूटर की आवश्यकता नहीं",
        "fr": "Aucun ordinateur quantique requis",
    },
    "solver.unknown_method": {
        "ja": "不明なソルバー手法: {method}",
        "en": "Unknown solver method: {method}",
        "zh": "未知的求解器方法: {method}",
        "ko": "알 수 없는 솔버 방법: {method}",
        "es": "Método de resolución desconocido: {method}",
        "hi": "अज्ञात सॉल्वर विधि: {method}",
        "fr": "Méthode de résolution inconnue : {method}",
    },
    "surprise.fit_first": {
        "ja": "先に fit() を呼んでください",
        "en": "Call fit() first",
        "zh": "请先调用 fit()",
        "ko": "먼저 fit()을 호출하세요",
        "es": "Llame a fit() primero",
        "hi": "पहले fit() कॉल करें",
        "fr": "Appelez fit() d'abord",
    },
    "surprise.unknown_method": {
        "ja": "不明なサプライズ手法: {method}",
        "en": "Unknown surprise method: {method}",
        "zh": "未知的惊喜方法: {method}",
        "ko": "알 수 없는 서프라이즈 방법: {method}",
        "es": "Método de sorpresa desconocido: {method}",
        "hi": "अज्ञात सरप्राइज़ विधि: {method}",
        "fr": "Méthode de surprise inconnue : {method}",
    },
    # -- Dharma engine messages --
    "dharma.supermodular": {
        "ja": "慈悲 (Karuna) = 超モジュラ関数 — 相乗効果",
        "en": "Karuna (Compassion) = supermodular function — synergy effect",
        "zh": "慈悲 (Karuna) = 超模函数 — 协同效应",
        "ko": "자비 (Karuna) = 초모듈러 함수 — 상승 효과",
        "es": "Karuna (Compasión) = función supermodular — efecto sinérgico",
        "hi": "करुणा (Karuna) = सुपरमॉड्यूलर फ़ंक्शन — सहक्रिया प्रभाव",
        "fr": "Karuna (Compassion) = fonction supermodulaire — effet de synergie",
    },
    "dharma.submodular": {
        "ja": "持戒 (Sila) = 劣モジュラ関数 — 限界効用逓減",
        "en": "Sila (Conduct) = submodular function — diminishing returns",
        "zh": "持戒 (Sila) = 子模函数 — 边际效用递减",
        "ko": "지계 (Sila) = 서브모듈러 함수 — 한계 효용 체감",
        "es": "Sila (Conducta) = función submodular — rendimientos decrecientes",
        "hi": "शील (Sila) = सबमॉड्यूलर फ़ंक्शन — ह्रासमान प्रतिफल",
        "fr": "Sila (Conduite) = fonction sous-modulaire — rendements décroissants",
    },
    "dharma.middle_way": {
        "ja": "中道 (Madhyamaka) = カオスの縁 (CV = 0.5)",
        "en": "Madhyamaka (Middle Way) = edge of chaos (CV = 0.5)",
        "zh": "中道 (Madhyamaka) = 混沌边缘 (CV = 0.5)",
        "ko": "중도 (Madhyamaka) = 혼돈의 가장자리 (CV = 0.5)",
        "es": "Madhyamaka (Camino Medio) = borde del caos (CV = 0.5)",
        "hi": "मध्यमक (मध्य मार्ग) = अराजकता का किनारा (CV = 0.5)",
        "fr": "Madhyamaka (Voie du Milieu) = bord du chaos (CV = 0.5)",
    },
    # -- Reasoning mode names --
    "mode.adaptive": {
        "ja": "応病与薬 — 器に合わせて変化する",
        "en": "Adaptive — medicine fit to illness",
        "zh": "应病与药 — 对症下药",
        "ko": "응병여약 — 병에 맞는 처방",
        "es": "Adaptivo — medicina ajustada a la enfermedad",
        "hi": "अनुकूली — रोग के अनुसार औषधि",
        "fr": "Adaptatif — remède adapté à la maladie",
    },
    "mode.theoretical": {
        "ja": "因明 — 仏教論理学",
        "en": "Theoretical — Buddhist formal logic (Hetuvidyā)",
        "zh": "因明 — 佛教逻辑学",
        "ko": "인명 — 불교 논리학",
        "es": "Teórico — lógica formal budista (Hetuvidyā)",
        "hi": "हेतुविद्या — बौद्ध तर्कशास्त्र",
        "fr": "Théorique — logique formelle bouddhiste (Hetuvidyā)",
    },
    "mode.hyper": {
        "ja": "般若の飛躍 — 直感的仮説生成",
        "en": "Abductive — Prajna's intuitive leap",
        "zh": "般若飞跃 — 直觉假设生成",
        "ko": "반야의 비약 — 직관적 가설 생성",
        "es": "Abductivo — salto intuitivo de Prajna",
        "hi": "प्रज्ञा की छलांग — अंतर्ज्ञानी परिकल्पना",
        "fr": "Abductif — saut intuitif de Prajna",
    },
    "mode.active": {
        "ja": "托鉢 — 外の真理を求める",
        "en": "Active Inference — seeking external truth",
        "zh": "托钵 — 寻求外在真理",
        "ko": "탁발 — 외부 진리 탐구",
        "es": "Inferencia activa — buscando la verdad externa",
        "hi": "पिण्डपात — बाह्य सत्य की खोज",
        "fr": "Inférence active — recherche de la vérité extérieure",
    },
    "mode.alaya": {
        "ja": "阿頼耶識 — 蔵識の記憶",
        "en": "Alaya Memory — storehouse consciousness",
        "zh": "阿赖耶识 — 藏识记忆",
        "ko": "아뢰야식 — 장식의 기억",
        "es": "Memoria Alaya — consciencia almacén",
        "hi": "आलय-विज्ञान — भंडार चेतना",
        "fr": "Mémoire Alaya — conscience entrepôt",
    },
    "mode.sleep": {
        "ja": "禅定 — 記憶の統合・圧縮",
        "en": "Sleep — NREM/REM memory consolidation",
        "zh": "禅定 — 记忆整合与压缩",
        "ko": "선정 — 기억 통합 및 압축",
        "es": "Sueño — consolidación de memoria NREM/REM",
        "hi": "ध्यान — NREM/REM स्मृति समेकन",
        "fr": "Sommeil — consolidation de mémoire NREM/REM",
    },
    "mode.embodied": {
        "ja": "六根 — 六つの感覚統合",
        "en": "Embodied — six sense modalities",
        "zh": "六根 — 六感统合",
        "ko": "육근 — 여섯 감각의 통합",
        "es": "Encarnado — seis modalidades sensoriales",
        "hi": "षड्-इन्द्रिय — छह इन्द्रिय विधाएँ",
        "fr": "Incarné — six modalités sensorielles",
    },
    "mode.pineal": {
        "ja": "松果体 — 量子意識",
        "en": "Pineal — quantum consciousness (hardware TRNG)",
        "zh": "松果体 — 量子意识",
        "ko": "송과체 — 양자 의식",
        "es": "Pineal — consciencia cuántica (TRNG por hardware)",
        "hi": "पीनियल — क्वांटम चेतना (हार्डवेयर TRNG)",
        "fr": "Pinéale — conscience quantique (TRNG matériel)",
    },
    # -- CLI messages --
    "cli.demo_running": {
        "ja": "デモ実行中...",
        "en": "Running demo...",
        "zh": "运行演示中...",
        "ko": "데모 실행 중...",
        "es": "Ejecutando demo...",
        "hi": "डेमो चल रहा है...",
        "fr": "Exécution de la démo...",
    },
    "cli.selected": {
        "ja": "選択されたインデックス: {indices}",
        "en": "Selected indices: {indices}",
        "zh": "选定的索引: {indices}",
        "ko": "선택된 인덱스: {indices}",
        "es": "Índices seleccionados: {indices}",
        "hi": "चयनित सूचकांक: {indices}",
        "fr": "Indices sélectionnés : {indices}",
    },
    # -- Server messages --
    "server.starting": {
        "ja": "自動降臨サーバー起動中",
        "en": "Starting auto-descent server",
        "zh": "自动降临服务器启动中",
        "ko": "자동 강림 서버 시작 중",
        "es": "Iniciando servidor de descenso automático",
        "hi": "स्वचालित अवतरण सर्वर शुरू हो रहा है",
        "fr": "Démarrage du serveur de descente automatique",
    },
    "server.api_key_required": {
        "ja": "Claude または Gemini の API キーが必要です",
        "en": "At least one API key (Claude or Gemini) is required",
        "zh": "至少需要一个 API 密钥（Claude 或 Gemini）",
        "ko": "최소 하나의 API 키(Claude 또는 Gemini)가 필요합니다",
        "es": "Se requiere al menos una clave API (Claude o Gemini)",
        "hi": "कम से कम एक API कुंजी (Claude या Gemini) आवश्यक है",
        "fr": "Au moins une clé API (Claude ou Gemini) est requise",
    },
    # -- Warnings --
    "warn.brute_force_knn": {
        "ja": "n={n:,} でブルートフォース k-NN は O(n²) です。hnswlib をインストールしてください: pip install hnswlib",
        "en": "Brute-force k-NN with n={n:,} is O(n²). Install hnswlib for O(n*log(n)): pip install hnswlib",
        "zh": "n={n:,} 的暴力 k-NN 是 O(n²)。安装 hnswlib 以获得 O(n*log(n)): pip install hnswlib",
        "ko": "n={n:,}에서 브루트포스 k-NN은 O(n²)입니다. O(n*log(n))을 위해 hnswlib를 설치하세요: pip install hnswlib",
        "es": "k-NN por fuerza bruta con n={n:,} es O(n²). Instale hnswlib: pip install hnswlib",
        "hi": "n={n:,} के साथ ब्रूट-फ़ोर्स k-NN O(n²) है। hnswlib इंस्टॉल करें: pip install hnswlib",
        "fr": "k-NN par force brute avec n={n:,} est O(n²). Installez hnswlib : pip install hnswlib",
    },
    # -- Ising SA --
    "warn.hnswlib_missing": {
        "ja": "hnswlib が見つかりません。ランダム射影近似 k-NN にフォールバックします",
        "en": "hnswlib not found. Falling back to random-projection approximate k-NN",
        "zh": "未找到 hnswlib。回退到随机投影近似 k-NN",
        "ko": "hnswlib을 찾을 수 없습니다. 랜덤 프로젝션 근사 k-NN으로 대체합니다",
        "es": "hnswlib no encontrado. Usando k-NN aproximado por proyección aleatoria",
        "hi": "hnswlib नहीं मिला। रैंडम प्रोजेक्शन अनुमानित k-NN पर वापस जा रहा है",
        "fr": "hnswlib introuvable. Repli sur k-NN approximatif par projection aléatoire",
    },
}


def set_locale(locale: str) -> None:
    """Set the global locale for LMM messages."""
    global _LOCALE
    _LOCALE = locale


def get_locale() -> str:
    """Get the current locale."""
    return _LOCALE


def t(key: str, **kwargs) -> str:
    """Translate a message key to the current locale.

    Falls back to English, then to the key itself if no translation found.
    """
    msg_dict = _MESSAGES.get(key)
    if msg_dict is None:
        return key
    msg = msg_dict.get(_LOCALE) or msg_dict.get("en", key)
    if kwargs:
        try:
            return msg.format(**kwargs)
        except (KeyError, IndexError):
            return msg
    return msg


def available_locales() -> list[str]:
    """Return all supported locale codes."""
    return ["ja", "en", "zh", "ko", "es", "hi", "fr"]
