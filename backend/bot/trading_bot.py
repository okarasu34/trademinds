import asyncio
from loguru import logger
from sqlalchemy import select
from datetime import datetime

from db.database import AsyncSessionLocal
from db.models import BotConfig, Strategy, BotStatus

class TradingBot:
    def __init__(self, user_id: str):
        self.user_id = user_id
        logger.info(f"TradingBot initialized for user: {self.user_id}")

    async def _get_active_strategies(self, db):
        """Veritabanından aktif stratejileri çeker."""
        try:
            result = await db.execute(select(Strategy).where(Strategy.is_active == True))
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Strateji çekme hatası: {e}")
            return []

    async def start(self, config=None):
        """API ve Scheduler'ın çağırdığı ana giriş noktası."""
        logger.info(f"Bot start sequence initiated for {self.user_id}")
        
        async with AsyncSessionLocal() as db:
            try:
                # Eğer config gelmediyse DB'den tazele
                if config is None:
                    result = await db.execute(
                        select(BotConfig).where(BotConfig.user_id == self.user_id)
                    )
                    config = result.scalars().first()

                if not config or config.status != BotStatus.RUNNING:
                    logger.warning(f"Bot {self.user_id} RUNNING değil, işlem iptal.")
                    return

                # Ana tarama ve işlem mantığını başlat
                await self._scan_and_execute(db, config)

            except Exception as e:
                logger.error(f"Bot start error: {e}")

    async def _scan_and_execute(self, db, config):
        """Stratejileri tarayan ve analiz yapan iç motor."""
        strategies = await self._get_active_strategies(db)
        if not strategies:
            logger.info("Aktif strateji yok.")
            return

        for strategy in strategies:
            logger.info(f"Analiz ediliyor: {strategy.name}")
            # Analiz ve trade mantığını buraya ekleyebilirsin
