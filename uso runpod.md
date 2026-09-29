CONTEXTO: Tenés acceso a las tools MCP de RunPod (mcp__devmcp__runpod_*). Otro Claude ya corrió una campaña completa de entrenamiento en GPU con estas mismas tools sin problemas. Estos son los pasos y trampas reales que encontró — seguilos en orden.

=== 1. ELEGIR GPU (antes de crear el pod) ===
- runpod_list_datacenters / runpod_list_gpu_types / runpod_list_affordable_gpus primero.
- NO uses H100 por default. Es $3.49/h y para modelos chicos (~60-100M params) sobra por mucho — VRAM usada real fue ~7-8GB de 80GB. Elegí por cómputo necesario, no por hábito.
- Recomendado para cargas chicas: RTX 3090 community (~$0.22/h, arquitectura sm_86, compatible con torch 2.4).
- Si usás un Network Volume (persistente): es SECURE y fija el pod a UN datacenter. Algunas GPUs (ej RTX 3090) solo existen como community pods y pueden no estar en ese datacenter — chequeá disponibilidad antes de atar el volumen.
- Compatibilidad de arquitectura: RTX PRO 6000 Blackwell = sm_120, necesita torch >=2.7 (falla con 2.4). H100 = sm_90, anda con torch 2.4. Verificá esto ANTES de crear el pod si el proyecto pinea una versión de torch.

=== 2. CREAR Y CONECTAR ===
- runpod_create_pod -> runpod_wait_for_pod_ssh (esperá esto SIEMPRE antes del primer exec, si no vas a pegar contra sshd que todavía no está arriba).
- Primer exec: verificá GPU con nvidia-smi y torch.cuda.is_available() antes de asumir nada.

=== 3. TRAER CÓDIGO/DATOS AL POD ===
- runpod_pod_upload sirve para archivos CHICOS (scripts, configs). Para archivos de varios GB SE CUELGA/TIMEOUTEA a los 60-90s pase lo que pase el timeoutSec que le pongas. NO insistas con upload en archivos grandes.
- Para datasets/checkpoints grandes: descargalos DIRECTO en el pod (huggingface_hub, wget, curl) en vez de subirlos desde tu máquina. Verificá sha256 en el pod contra el sha256 esperado ANTES de confiar en el archivo.

=== 4. EJECUTAR COMANDOS (el punto que más traba) ===
- runpod_pod_exec tiene timeout DURO de ~30s en la llamada MCP. Si tu comando (entrenar, instalar deps pesadas, cargar modelo grande) tarda más, la LLAMADA va a tirar timeout — pero el proceso puede haber arrancado igual si lo lanzaste con nohup.
- Patrón correcto para tareas largas:
  cd /workspace/proyecto && source venv/bin/activate && export PYTHONUNBUFFERED=1 && nohup python -u script.py > log.txt 2>&1 &
  echo "pid $!"
  Si ESTA llamada también da timeout, NO la repitas de inmediato — puede duplicar el proceso. Esperá unos segundos y poleá con exec CORTOS: `tail -n 30 log.txt` + `ps aux | grep -c "[s]cript.py"`.
- Siempre `python -u` + PYTHONUNBUFFERED=1, si no el print() queda bufferizado en el pipe/log y no ves progreso real.
- NUNCA hagas `pkill -f <patron>` para matar un proceso — si el patrón matchea también la línea de comando de la shell que está ejecutando el exec, te matás a vos mismo (el exec que estás corriendo). Matá por PID explícito.

=== 5. DESCARGAR RESULTADOS ===
- runpod_pod_download baja archivo por archivo (SFTP). Para una carpeta con varios archivos chicos: primero `tar czf salida.tgz carpeta/` en el pod, después descargás el único tarball.
- Verificá sha256 en AMBOS lados (pod y local) después de cada transferencia — upload y download. No asumas que llegó bien.

=== 6. CIERRE (obligatorio, no opcional) ===
- runpod_delete_pod apenas termines de descargar todo y verificar los sha256. El pod sigue facturando mientras exista, esté "idle" o no.
- runpod_list_pods para confirmar que no quedó nada corriendo por error.

Seguí este orden y no deberías trabarte. Si algo del entorno (imagen, versión de torch/cuda) no calza con lo que el proyecto necesita, es mejor cambiar la imagen del pod ANTES de instalar nada, no parchear después.
