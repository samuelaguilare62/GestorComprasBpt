import os
import logging
import json
import re
import asyncio
from datetime import datetime
import easyocr
import cv2
import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import tempfile

# Configurar logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Variables de entorno
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
ALLOWED_USERS = [int(id) for id in os.getenv('ALLOWED_USERS', '').split(',') if id]

# SOLUCIÓN: Forzar la descarga de modelos al inicio
def initialize_easyocr():
    """Inicializa EasyOCR descargando los modelos explícitamente"""
    try:
        # Descargar modelos explícitamente
        logging.info("Descargando modelos de EasyOCR...")
        reader = easyocr.Reader(
            ['es', 'en'], 
            gpu=False,
            download_enabled=True,
            model_storage_directory='/tmp/.easyocr'  # Usar directorio temporal persistente
        )
        logging.info("Modelos de EasyOCR cargados correctamente")
        return reader
    except Exception as e:
        logging.error(f"Error inicializando EasyOCR: {e}")
        raise

class GestorCompras:
    def __init__(self):
        try:
            self.reader = initialize_easyocr()
            self.data_file = 'compras.json'
            self.load_data()
            logging.info("Gestor de Compras inicializado correctamente")
        except Exception as e:
            logging.error(f"Error en inicialización: {e}")
            self.reader = None
    
    def load_data(self):
        """Carga los datos existentes"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            else:
                self.data = {'compras': []}
        except Exception as e:
            logging.error(f"Error cargando datos: {e}")
            self.data = {'compras': []}
    
    def save_data(self):
        """Guarda los datos"""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Error guardando datos: {e}")
    
    def preprocess_image(self, image_path):
        """Mejora la imagen para mejor OCR"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("No se pudo cargar la imagen")
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            denoised = cv2.medianBlur(gray, 5)
            thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            return thresh
        except Exception as e:
            logging.error(f"Error procesando imagen: {e}")
            return None
    
    def extract_text_from_ticket(self, image_path):
        """Extrae texto de la imagen del ticket"""
        try:
            if self.reader is None:
                return "Error: OCR no inicializado", []
                
            processed_image = self.preprocess_image(image_path)
            if processed_image is None:
                return "Error procesando imagen", []
                
            results = self.reader.readtext(processed_image, detail=1)
            full_text = ' '.join([result[1] for result in results])
            return full_text, results
        except Exception as e:
            logging.error(f"Error en OCR: {e}")
            return f"Error en OCR: {str(e)}", []
    
    def parse_ticket_info(self, text):
        """Extrae información específica del ticket"""
        info = {
            'fecha': None,
            'hora': None,
            'total': None,
            'establecimiento': None,
            'productos': []
        }
        
        try:
            # Buscar fecha (formato dd/mm/aaaa o dd-mm-aaaa)
            fecha_pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
            fecha_match = re.search(fecha_pattern, text)
            if fecha_match:
                info['fecha'] = fecha_match.group(1)
            
            # Buscar hora (formato hh:mm)
            hora_pattern = r'(\d{1,2}:\d{2})'
            hora_match = re.search(hora_pattern, text)
            if hora_match:
                info['hora'] = hora_match.group(1)
            
            # Buscar total (patrones comunes)
            total_patterns = [
                r'total[\s:]*\$?\s*(\d+[.,]\d+)',
                r'total[\s:]*\$?\s*(\d+)',
                r'importe[\s:]*\$?\s*(\d+[.,]\d+)',
                r'[\$€]\s*(\d+[.,]\d+)',
                r'final[\s:]*\$?\s*(\d+[.,]\d+)'
            ]
            
            for pattern in total_patterns:
                total_match = re.search(pattern, text.lower())
                if total_match:
                    info['total'] = total_match.group(1).replace(',', '.')
                    break
            
            # Buscar establecimiento
            lines = text.split('\n')
            for line in lines[:3]:
                if len(line.strip()) > 5 and not any(word in line.lower() for word in ['total', 'fecha', 'hora', 'ticket']):
                    info['establecimiento'] = line.strip()[:50]
                    break
            
            return info
            
        except Exception as e:
            logging.error(f"Error parseando ticket: {e}")
            return info
    
    def add_purchase(self, ticket_info, image_path):
        """Añade una nueva compra"""
        try:
            compra = {
                'id': len(self.data['compras']) + 1,
                'fecha_compra': ticket_info.get('fecha'),
                'hora_compra': ticket_info.get('hora'),
                'establecimiento': ticket_info.get('establecimiento'),
                'total': float(ticket_info.get('total', 0)) if ticket_info.get('total') else 0,
                'productos': ticket_info.get('productos', []),
                'imagen_ticket': image_path,
                'fecha_registro': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.data['compras'].append(compra)
            self.save_data()
            return compra
        except Exception as e:
            logging.error(f"Error añadiendo compra: {e}")
            return None
    
    def get_stats(self):
        """Obtiene estadísticas de compras"""
        try:
            if not self.data['compras']:
                return None
            
            total_compras = len(self.data['compras'])
            gasto_total = sum(compra['total'] for compra in self.data['compras'])
            compra_promedio = gasto_total / total_compras if total_compras > 0 else 0
            
            establecimientos = [compra['establecimiento'] for compra in self.data['compras'] if compra['establecimiento']]
            establecimiento_frecuente = max(set(establecimientos), key=establecimientos.count) if establecimientos else 'N/A'
            
            return {
                'total_compras': total_compras,
                'gasto_total': gasto_total,
                'compra_promedio': compra_promedio,
                'establecimiento_frecuente': establecimiento_frecuente
            }
        except Exception as e:
            logging.error(f"Error calculando stats: {e}")
            return None

class TelegramBot:
    def __init__(self, gestor_compras):
        self.gestor = gestor_compras
        self.application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        
        # Handlers
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("stats", self.stats))
        self.application.add_handler(CommandHandler("compras", self.list_compras))
        self.application.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        
        if user_id not in ALLOWED_USERS:
            await update.message.reply_text("❌ No tienes acceso a este bot.")
            return
        
        welcome_text = """
🛒 *Gestor de Compras*

Envía una foto de tu ticket y extraeré automáticamente:
• 📅 Fecha y hora
• 🏪 Establecimiento  
• 💰 Total de la compra

*Comandos:*
/start - Este mensaje
/stats - Estadísticas
/compras - Últimas compras
"""
        await update.message.reply_text(welcome_text, parse_mode='Markdown')
    
    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        
        if user_id not in ALLOWED_USERS:
            await update.message.reply_text("❌ No tienes acceso.")
            return
        
        await update.message.reply_text("📷 Procesando ticket...")
        
        try:
            # Descargar foto
            photo_file = await update.message.photo[-1].get_file()
            
            # Usar archivo temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                image_path = tmp_file.name
            
            await photo_file.download_to_drive(image_path)
            
            # Procesar con OCR
            text, _ = self.gestor.extract_text_from_ticket(image_path)
            
            if "Error" in text:
                await update.message.reply_text("❌ Error procesando imagen. Intenta con otra foto.")
                return
                
            ticket_info = self.gestor.parse_ticket_info(text)
            
            # Guardar compra
            compra = self.gestor.add_purchase(ticket_info, "ticket_temp.jpg")
            
            # Respuesta
            response = f"""
✅ *Ticket procesado*

🏪 *Lugar:* {ticket_info.get('establecimiento', 'No identificado')}
📅 *Fecha:* {ticket_info.get('fecha', 'No identificada')}
⏰ *Hora:* {ticket_info.get('hora', 'No identificada')}
💰 *Total:* ${ticket_info.get('total', '0')}
"""
            await update.message.reply_text(response, parse_mode='Markdown')
            
            # Limpiar archivo temporal
            os.unlink(image_path)
            
        except Exception as e:
            logging.error(f"Error procesando foto: {e}")
            await update.message.reply_text("❌ Error procesando ticket. Intenta con otra foto.")
    
    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        
        if user_id not in ALLOWED_USERS:
            await update.message.reply_text("❌ No tienes acceso.")
            return
        
        stats = self.gestor.get_stats()
        
        if not stats:
            await update.message.reply_text("📊 Aún no hay compras.")
            return
        
        stats_text = f"""
📊 *Estadísticas*

• 🛒 Compras: {stats['total_compras']}
• 💵 Gasto total: ${stats['gasto_total']:.2f}
• 📈 Promedio: ${stats['compra_promedio']:.2f}
• 🏪 Lugar frecuente: {stats['establecimiento_frecuente']}
"""
        await update.message.reply_text(stats_text, parse_mode='Markdown')
    
    async def list_compras(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        
        if user_id not in ALLOWED_USERS:
            await update.message.reply_text("❌ No tienes acceso.")
            return
        
        compras = self.gestor.data['compras'][-5:]
        
        if not compras:
            await update.message.reply_text("📝 No hay compras.")
            return
        
        compras_text = "📝 *Últimas compras:*\n\n"
        for compra in compras:
            compras_text += f"🏪 {compra['establecimiento']}\n"
            compras_text += f"📅 {compra['fecha_compra']} - 💰 ${compra['total']}\n"
            compras_text += "─" * 20 + "\n"
        
        await update.message.reply_text(compras_text, parse_mode='Markdown')
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("📷 Envía una foto de ticket o usa /stats /compras")
    
    def run(self):
        """Inicia el bot"""
        logging.info("Iniciando bot de Telegram...")
        self.application.run_polling()

def main():
    logging.info("🚀 Iniciando Gestor de Compras...")
    
    # Verificar variables de entorno
    if not TELEGRAM_BOT_TOKEN:
        logging.error("❌ TELEGRAM_BOT_TOKEN no configurado")
        return
    
    try:
        gestor = GestorCompras()
        bot = TelegramBot(gestor)
        bot.run()
    except Exception as e:
        logging.error(f"❌ Error crítico: {e}")

if __name__ == "__main__":
    main()
