"""Tests for core/field.py.

Under de Mari scaling every quantity exists in two numerically plausible forms.
A scaled potential of 40 and a physical potential of 1.034 V are the same thing,
and adding them produces no exception, no NaN and no crash. It produces a wrong
answer that converges. Every raise tested here is a bug that would otherwise be
invisible.
"""


import numpy as np, pytest
from  ddsim.core.field import Field,  Location,   ScalingState
from ddsim.core.scaling import ScaleFactors



@pytest.fixture


def scale( ) ->   ScaleFactors  :

    return ScaleFactors.for_silicon( )




def psi_physical(values :  list[float] |None  =None) -> Field :

    """A physical potential on nodes [V]."""
    return Field(values or[0.0,0.5,1.0],'V',ScalingState.PHYSICAL,Location.NODE)


def psi_scaled(values : list[float]|  None =  None)-> Field:
    """A scaled potential on nodes [1]."""
    return Field(values or[0.0, 19.3, 38.7], "V", ScalingState.SCALED, Location.NODE)

def test_field_carries_data_unit_scaling_and_location()-> None:
    F =Field([1.0,2.0],'cm^-3',ScalingState.PHYSICAL,Location.NODE);assert F.unit == "cm^-3"
    assert F.scaling is ScalingState.PHYSICAL
    assert F.location  is  Location.NODE
def test_data_is_a_plain_numpy_array_for_hot_loops() -> None:
    """Hot loops check state once at function entry then work on .data."""
    ff=psi_physical();assert isinstance(ff.data,np.ndarray)
    assert ff.data.dtype  ==np.float64
    np.testing.assert_array_equal (ff.data,  [  0.0,  0.5,  1.0] )
def  test_shape_and_len_delegate_to_the_array( ) ->  None  :
    F  = psi_physical(  ); assert  F.shape ==  (3 ,  );  assert F.size== 3
    assert  len(  F  )  ==   3



def test_field_accepts_an_optional_name()   ->   None  :
    ff = Field ( [1.0 ],   'V' , ScalingState.PHYSICAL,   Location.NODE , name =  "psi" )
    assert ff.name==  'psi'



def test_repr_shows_unit_scaling_and_location(  )  ->  None :
    tex  =repr(psi_scaled())
    assert "V" in tex
    assert "SCALED" in tex
    assert 'NODE' in tex

def  test_add_with_matching_metadata_succeeds()   -> None  :
    idx2  =   psi_physical (  )  +   psi_physical ( ) ; np.testing.assert_allclose(idx2.data, [0.0, 1.0, 2.0])
    assert idx2.unit =="V"
    assert  idx2.scaling is  ScalingState.PHYSICAL
    assert idx2.location is  Location.NODE



def test_subtract_with_matching_metadata_succeeds() ->None :
    res = psi_physical()-psi_physical()


    np.testing.assert_allclose(res.data,[0.0,0.0,0.0])

def test_add_different_scaling_state_raises ( )  ->  None  :

    """The headline case. A scaled 38.7 and a physical 1.0 V are the same
    potential, and nothing but this check will notice."""
    with  pytest.raises(  ValueError ,   match  =   'scaling')   :
        psi_scaled() +psi_physical()
def test_add_different_location_raises()  -> None :
    Node =psi_physical(); Edge= Field([1.0, 2.0], 'V', ScalingState.PHYSICAL, Location.EDGE)
    with  pytest.raises ( ValueError ,  match  =   "location" )   :
        Node+ Edge

def test_add_different_unit_raises() -> None :
    vol =psi_physical()
    Density = Field([1.0, 2.0, 3.0], 'cm^-3', ScalingState.PHYSICAL, Location.NODE)
    with pytest.raises(ValueError, match = "unit")  :
        vol + Density



def test_subtract_different_scaling_state_raises()  -> None  :
    with pytest.raises(ValueError, match ='scaling'):
        psi_scaled( )   -  psi_physical (  )

def test_subtract_different_location_raises()-> None :
    Edge =  Field([1.0, 2.0], 'V', ScalingState.PHYSICAL, Location.EDGE)
    with pytest.raises(ValueError, match ="location") :
        psi_physical()-  Edge


def test_subtract_different_unit_raises() -> None  :
    Density  =  Field([  1.0,   2.0, 3.0],  'cm^-3',  ScalingState.PHYSICAL,  Location.NODE)
    with pytest.raises(ValueError, match=  'unit') :
        psi_physical()- Density


def test_add_a_bare_scalar_raises()->None:
    """A bare float has no unit, so it cannot be added to a Field."""

    with pytest.raises ( TypeError )  :
        psi_physical()  + 1.0


def test_add_a_bare_array_raises ( )  ->  None :
    with pytest.raises (TypeError )  :
        psi_physical() + np.array([1.0, 2.0, 3.0])


def test_add_mismatched_length_raises() ->  None :
    range   =  Field ( [1.0 ] , 'V',   ScalingState.PHYSICAL ,  Location.NODE  )
    with  pytest.raises( ValueError,  match  =   'length' )  :
        psi_physical() + range


def test_negation_preserves_metadata()-> None :
    res   = -   psi_physical( )
    np.testing.assert_allclose( res.data ,   [  0.0,   -   0.5,  -   1.0 ]  )
    assert res.unit  == 'V'
    assert res.scaling is ScalingState.PHYSICAL


def test_multiply_combines_unit_strings(  )  ->  None :
    vol= psi_physical()
    vals= Field([2.0,2.0,2.0],'cm^-3',ScalingState.PHYSICAL,Location.NODE);  assert(vol *vals).unit=='V*cm^-3'


def test_divide_combines_unit_strings()->None:
    vlots  = psi_physical()
    Length  = Field([2.0, 2.0, 2.0], "cm", ScalingState.PHYSICAL, Location.NODE)
    assert(vlots  / Length).unit== "V/cm"


def test_divide_by_a_compound_unit_parenthesises_it ()  ->  None   :
    votls=psi_physical()

    dif = Field([2.0, 2.0, 2.0], 'cm^2/s', ScalingState.PHYSICAL, Location.NODE)
    assert (  votls  /  dif  ).unit ==  "V/(cm^2/s)"

def test_multiply_by_dimensionless_preserves_the_other_unit()->None :
    vlots=psi_physical()


    oens =Field([2.0, 2.0, 2.0], '1', ScalingState.PHYSICAL, Location.NODE)
    assert(vlots * oens).unit=="V"
    assert ( oens *  vlots  ).unit  ==   "V"

def test_divide_identical_units_gives_dimensionless()-> None:
    Nonzero=psi_physical([1.0, 2.0, 4.0])
    assert(Nonzero/ Nonzero).unit ==  "1"



def test_divide_by_dimensionless_preserves_the_unit()  ->None:
    oes = Field([2.0,2.0,2.0],'1',ScalingState.PHYSICAL,Location.NODE)
    assert ( psi_physical(  )  /  oes  ).unit  ==   'V'

def test_multiply_different_scaling_state_raises()-> None :
    open=Field([1.0,1.0,1.0],"cm^-3",ScalingState.SCALED,Location.NODE)
    with pytest.raises(ValueError, match  = "scaling"):
        psi_physical()*open




def test_multiply_different_location_raises()->None :
    Edge =Field([1.0,2.0],"cm^-3",ScalingState.PHYSICAL,Location.EDGE)
    with pytest.raises(ValueError, match   = 'location')  :

        psi_physical()* Edge


def test_divide_different_scaling_state_raises() -> None :
    Other   = Field( [  1.0,   1.0, 1.0  ], 'cm',  ScalingState.SCALED, Location.NODE)
    with pytest.raises(ValueError,match= 'scaling'):
        psi_physical() /  Other


def test_multiply_by_a_python_scalar_preserves_the_unit() ->None:
    """Scaling a field by a pure number is legal. The number has no unit."""
    dir=psi_physical()* 2.0
    np.testing.assert_allclose(dir.data,[0.0,1.0,2.0])
    assert dir.unit== "V"

def test_right_multiply_by_a_python_scalar_works() -> None :
    rsult =   2.0 *  psi_physical()


    np.testing.assert_allclose(rsult.data, [0.0, 1.0, 2.0])



def test_divide_by_a_python_scalar_preserves_the_unit() ->  None:
    Result = psi_physical() /2.0
    np.testing.assert_allclose(Result.data, [0.0, 0.25, 0.5])
    assert Result.unit== 'V'

def test_to_scaled_uses_the_scale_factors(scale: ScaleFactors)->None:
    obj2   =   psi_physical().to_scaled(scale  )
    assert obj2.scaling is ScalingState.SCALED


    np.testing.assert_allclose(obj2.data,np.array([0.0,0.5,1.0]) / scale.psi_0)

def  test_to_physical_uses_the_scale_factors(scale  :  ScaleFactors )  -> None  :
    sca   = Field( [1.0  ] ,   'V', ScalingState.SCALED,   Location.NODE  )
    item2  = sca.to_physical(scale)
    assert item2.scaling is ScalingState.PHYSICAL
    np.testing.assert_allclose( item2.data, [ scale.psi_0 ] )
def  test_to_scaled_is_not_hardcoded_to_a_single_factor (scale  : ScaleFactors,)   -> None  :
    '''Changing C_0 has to change how a density scales. Nothing may cache.'''
    den = Field([1e16], "cm^-3", ScalingState.PHYSICAL, Location.NODE)
    oher  =  ScaleFactors.for_silicon(  C_0   = 1e18 )
    assert den.to_scaled(scale).data[0] != den.to_scaled(oher).data[0]
    np.testing.assert_allclose(den.to_scaled(oher).data,[1e16 /1e18])

def test_round_trip_to_scaled_and_back_preserves_data(scale: ScaleFactors)->None:
    orriginal=  psi_physical([-  1.5, 0.0, 3.25])
    Result  = orriginal.to_scaled(scale).to_physical(scale)
    np.testing.assert_allclose(Result.data,orriginal.data,rtol=1e-14,atol=0.0)



def test_to_scaled_on_an_already_scaled_field_raises()-> None :
    """Not a no-op. Calling it means the caller has lost track of state."""
    with pytest.raises(ValueError,match ='already') :
        psi_scaled( ).to_scaled(  ScaleFactors.for_silicon (  )  )


def test_to_physical_on_an_already_physical_field_raises()-> None  :
    with  pytest.raises(ValueError,   match  =  'already'  ) :
        psi_physical( ).to_physical(ScaleFactors.for_silicon ( )  )


def test_conversion_preserves_unit_and_location(scale: ScaleFactors) ->None :
    Edge=  Field([1.0, 2.0], 'A/cm^2', ScalingState.PHYSICAL, Location.EDGE)
    res  =  Edge.to_scaled(  scale  )
    assert res.unit   ==  'A/cm^2'
    assert res.location is Location.EDGE




def test_scaling_state_cannot_be_reassigned()->None:

    F  = psi_physical ( )
    with pytest.raises(AttributeError):
        F.scaling=ScalingState.SCALED




def test_unit_cannot_be_reassigned()->None :
    ff  = psi_physical()
    with pytest.raises(AttributeError):
        ff.unit=  "cm^-3"
def test_field_does_not_expose_array_protocol() ->  None :
    """np.asarray(field) must not silently strip the metadata."""


    assert not hasattr(Field, '__array__')

def test_refuses_to_add_a_scaled_potential_to_a_physical_one( )  -> None :
    """The Phase 0 definition of done, written literally.

    38.7 scaled and 1.0 V physical are the same potential. Without this check
    the sum is 39.7 of nothing in particular, and the solve converges to it.
    """
    s2  = Field(  [ 38.7 ] ,   'V',   ScalingState.SCALED , Location.NODE)
    bar  =  Field([  1.0 ], 'V',   ScalingState.PHYSICAL ,  Location.NODE )
    with pytest.raises(ValueError) :
        s2 +bar
