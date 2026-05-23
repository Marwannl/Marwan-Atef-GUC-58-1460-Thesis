from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import Base
from models import Stock, TrendRecord, Prediction, DailySnapshot

engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)


def setup_function():
    Base.metadata.create_all(bind=engine)


def teardown_function():
    Base.metadata.drop_all(bind=engine)


def test_stock_table_creates():
    db = Session()
    s = Stock(ticker="AAPL", company_name="Apple Inc.", sector="Technology")
    db.add(s)
    db.commit()
    assert db.query(Stock).filter_by(ticker="AAPL").first().company_name == "Apple Inc."
    db.close()


def test_daily_snapshot_creates():
    db = Session()
    snap = DailySnapshot(
        ticker="AAPL", date="2026-04-01",
        open=170.0, high=175.0, low=169.0, close=174.0, volume=50000000
    )
    db.add(snap)
    db.commit()
    assert db.query(DailySnapshot).first().close == 174.0
    db.close()


def test_trend_record_creates():
    db = Session()
    t = TrendRecord(
        ticker="AAPL", direction="up", started_at="2026-03-01",
        start_price=170.0, reversal_confirmed=False
    )
    db.add(t)
    db.commit()
    assert db.query(TrendRecord).first().direction == "up"
    db.close()


def test_prediction_creates():
    db = Session()
    p = Prediction(
        ticker="AAPL", trend_direction="up",
        predicted_duration_days=10,
        survival_prob_3d=0.81, survival_prob_5d=0.68, survival_prob_7d=0.54
    )
    db.add(p)
    db.commit()
    assert db.query(Prediction).first().survival_prob_7d == 0.54
    db.close()
